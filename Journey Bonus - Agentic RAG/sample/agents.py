from openai import AsyncAzureOpenAI, pydantic_function_tool, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials_async import AsyncTokenCredential
import chainlit as cl
from chainlit.context import ChainlitContextException
from typing import List, Any, Optional, Dict,Tuple
from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator
import prompty
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import MessageRole, BingGroundingTool
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt, retry_if_exception_type
from copy import deepcopy
import asyncio

import json
import functools
from dataclasses import dataclass

from dotenv import load_dotenv
from models import ActionList, Action, ActionBase, SearchSources, Document, AnswerSources, AnswerAction, EvaluationAction, EvaluatorResponse, FilterDocumentsAction, IndexSearchAction, QueryRewriteAction, QuestionAction, ReflectionAction, StopAction, ToolChoiceReflection, WebSearchAction, WebCitation


answer_prompt = prompty.load("prompts/chat_answer_question.prompty")
query_rewrite_prompt = prompty.load("prompts/chat_query_rewrite.prompty")
reflect_prompt = prompty.load("prompts/chat_reflect_answer.prompty")

@dataclass
class AppState:
    openai_client: AsyncAzureOpenAI
    search_client: SearchClient
    answer_generator_deployment: str
    query_rewriter_deployment: str
    reflection_deployment: str
    groundedness_evaluator: GroundednessEvaluator
    relevance_evaluator: RelevanceEvaluator
    ai_project_client: AIProjectClient
    bing_grounding_name: str
    bing_tool: Optional[BingGroundingTool] = None


def running_in_cl() -> bool:
    try:
        cl.context.session.thread_id
        return True
    except ChainlitContextException:
        return False

# Generate steps if running in chainlit, otherwise do nothing
def step(func):
    @functools.wraps(func)
    async def step_wrapper(*args, **kwargs):
        cl_step: Optional[cl.Step] = None if not running_in_cl() else cl.Step(name=func.__name__)

        if cl_step:
            async with cl_step:
                cl_step.input = kwargs
                output = await func(*args, **kwargs)
                cl_step.output = output.model_dump()
                if isinstance(output, ActionBase) and output.render_during_step():
                    await output.render()
                return output
        else:
            return await func(*args, **kwargs)
    
    return step_wrapper


@step
async def rewrite_query(app_state: AppState, question: str, message_history: List[ChatCompletionMessageParam]) -> QueryRewriteAction:
    query_messages = prompty.prepare(
        query_rewrite_prompt, {"user_query": question, "past_messages": message_history}
    )
    result = await app_state.openai_client.beta.chat.completions.parse(
        messages=query_messages,
        response_format=SearchSources,
        model=app_state.query_rewriter_deployment,
        temperature=0
    )
    if parsed := result.choices[0].message.parsed:
        return QueryRewriteAction(
            query=question, message_history=message_history, rewritten_query=parsed.search_query
        )

    return QueryRewriteAction(
        query=question, message_history=message_history, rewritten_query=None
    )

@step
async def filter_documents(documents: List[Document], threshold: float = 2) -> FilterDocumentsAction:
    filtered_documents = [document for document in documents if document.score >= threshold]
    return FilterDocumentsAction(
        threshold=threshold,
        documents=documents,
        filtered_documents=filtered_documents
    )

@step
async def index_search(app_state: AppState, question: str, message_history: List[ChatCompletionMessageParam]) -> IndexSearchAction:
    query = await rewrite_query(app_state, question, message_history)
    search_query = query.rewritten_query or question
    response = await app_state.search_client.search(
        search_text=search_query,
        vector_queries=[VectorizableTextQuery(text=search_query, k_nearest_neighbors=50, fields="embedding")],
        query_type="semantic",
        semantic_configuration_name="default",
        top=10,
        select=["content", "id", "sourcepage"]
    )
    results = [Document(content=result["content"], id=result["id"], score=result["@search.reranker_score"], title=result["sourcepage"]) async for result in response]
    filtered_results = await filter_documents(results)
    return IndexSearchAction(
        query=question,
        rewritten_query=query.rewritten_query,
        documents=filtered_results.filtered_documents,
    )

@step
async def web_search(app_state: AppState, query: str) -> WebSearchAction:
    # Initialize agent bing tool and add the connection id
    if not app_state.bing_tool:
        bing_connection = await app_state.ai_project_client.connections.get(app_state.bing_grounding_name)
        app_state.bing_tool = BingGroundingTool(connection_id=bing_connection.id)
    search_agent = await app_state.agents.create_agent(
        model="gpt-4o",
        name="my-web-assistant",
        instructions="You are a helpful assistant",
        tools=app_state.bing_tool.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    thread = await app_state.ai_project_client.agents.create_thread()
    await app_state.ai_project_client.agents.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Search the web: " + query,
    )
    async for run_attempt in AsyncRetrying(retry=retry_if_exception_type(ValueError), wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10)):
        with run_attempt:
            run = await app_state.ai_project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=search_agent.id)
            if run.status == "failed" and run.last_error and run.last_error.get('code') == 'rate_limit_exceeded':
                raise ValueError("Rate limit exceeded. Please try again later.")

            if run.status == "failed":
                raise ValueError(run.last_error)
        
            response_messages = await app_state.ai_project_client.agents.list_messages(thread_id=thread.id)
            response_message = response_messages.get_last_message_by_role(
                MessageRole.AGENT
            )
            if response_message.text_messages and len(response_message.text_messages) > 0:
                response_content = response_message.text_messages[0].text.value
                citations = [ WebCitation(name=annotation.text, url=annotation.url_citation.url, title=annotation.url_citation.title) for annotation in response_message.url_citation_annotations ]
            else:
                response_content = ""
                citations = []

        
            return WebSearchAction(
                query=query,
                response_content=response_content,
                citations=citations,
            )

@step
async def generate_answer(app_state: AppState, question: str, answer_sources: AnswerSources, previous_answer: Optional[str] = None, previous_answer_reflection: Optional[str] = None, previous_answer_evaluations: Optional[List[EvaluatorResponse]] = None ) -> AnswerAction:    
    messages = prompty.prepare(
        answer_prompt,
        {
            "user_query": question,
            "context": answer_sources.create_context(),
            "previous_answer_reflection": previous_answer_reflection,
            "previous_answer": previous_answer,
            "previous_answer_evaluations": previous_answer_evaluations
        }
    )

    stream: AsyncStream[ChatCompletionChunk] = await app_state.openai_client.chat.completions.create(
        messages=messages,
        stream=True,
        model=app_state.answer_generator_deployment,
        temperature=0.3
    )

    answer_text = ""
    msg: Optional[cl.Message] = None if not running_in_cl() else cl.Message("")
    async for part in stream:
        if part.choices and (token := part.choices[0].delta.content or ""):
            answer_text += token
            if msg:
                await msg.stream_token(token)
    
    
    result = AnswerAction(
        question_text=question,
        answer_text=answer_text,
        answer_sources=answer_sources,
    )

    if msg:
        msg.elements = result.create_pdf_citations()
        await msg.update()
    
    return result

@step
async def evaluate_answer(app_state: AppState, answer: AnswerAction, previous_reflection: Optional[ReflectionAction] = None) -> EvaluationAction:
    evaluator_context = answer.answer_sources.create_context()
    if previous_reflection:
        evaluator_context += f"\n\nPrevious Answer: {previous_reflection.answer_text}\n\nPrevious Answer Incorrect Reason: {previous_reflection.answer_reflection}"

    def eval_groundedness() -> Dict[str, Any]:
        return app_state.groundedness_evaluator(
            **{
                "query": answer.question_text,
                "context": evaluator_context,
                "response": answer.answer_text
            }
        )
    async_eval_groundedness = cl.make_async(eval_groundedness)

    def eval_relevance() -> Dict[str, Any]:
        return app_state.relevance_evaluator(
            **{
                "query": answer.question_text,
                "response": answer.answer_text
            }
        )
    async_eval_relevance = cl.make_async(eval_relevance)

    results = await asyncio.gather(async_eval_groundedness(), async_eval_relevance())
    groundedness_result = results[0]
    relevance_result = results[1]

    results = [
        EvaluatorResponse(
            type="relevance",
            score=relevance_result['relevance'],
            explanation=relevance_result['relevance_reason']
        ),
        EvaluatorResponse(
            type="groundedness",
            score=groundedness_result['groundedness'],
            explanation=groundedness_result['groundedness_reason']
        )
    ]

    return EvaluationAction(
        evaluations=results,
        answer=answer
    )


@step
async def reflect_answer(app_state: AppState, action_list: ActionList) -> ReflectionAction:
    last_answer_action = action_list.get_last_answer()
    last_query = action_list.get_last_query()
    last_evaluation = action_list.get_last_evaluation()
    assert last_answer_action != None
    assert last_query != None
    assert last_evaluation != None
    reflect_messages = prompty.prepare(
        reflect_prompt,
        {
            "previous_answer": last_answer_action.answer_text,
            "previous_sources": last_answer_action.answer_sources.create_context(),
            "previous_query": last_query,
            "previous_answer_evlauation": "\n".join(f"{evaluation.type}: {evaluation.score}, {evaluation.explanation}" for evaluation in last_evaluation.evaluations)
        }
    )
    result: ChatCompletion = await app_state.openai_client.chat.completions.create(
        messages=reflect_messages,
        tools=[pydantic_function_tool(ToolChoiceReflection)],
        model=app_state.reflection_deployment
    )
    action = ReflectionAction(
        answer_text="I don't know",
        answer_reflection="I don't know",
        follow_up_action="stop",
        follow_up_action_reason="I don't know what to do next.",
        actions=action_list.actions
    )

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        tool_reflection = json.loads(tool_call.function.arguments)
        follow_up_action_str = "stop"
        if "index_query" in tool_reflection["follow_up_action"]:
            follow_up_action_str = "index_search"
            next_query = tool_reflection["follow_up_action"]["index_query"]
        elif "web_query" in tool_reflection["follow_up_action"]:
            follow_up_action_str = "web_search"
            next_query = tool_reflection["follow_up_action"]["web_query"]
        action = ReflectionAction(
            answer_text=last_answer_action.answer_text,
            answer_reflection=tool_reflection["answer_reflection"],
            follow_up_action=follow_up_action_str,
            follow_up_action_reason=tool_reflection["follow_up_action_reason"],
            next_query=next_query,
            actions=action_list.actions
        )

    return action


async def index_search_only(app_state: AppState, message: str, message_history: Optional[List[ChatCompletionMessageParam]] = None) -> ActionList:
    question_action = QuestionAction(question_text=message)
    index_search_action = await index_search(app_state, message, message_history)
    answer_action = await generate_answer(app_state, message, AnswerSources.from_index_search(message_history, index_search_action))

    return ActionList(actions=[question_action, index_search_action, answer_action])

async def reflect_next_action(app_state: AppState, reflection: ReflectionAction, answer_sources: AnswerSources) -> Tuple[Action, AnswerSources]:
    if reflection.follow_up_action == "index_search" and reflection.next_query:
        index_action = await index_search(app_state, reflection.next_query, answer_sources.message_history)
        next_sources = answer_sources.add_index_search(index_action)
        return (index_action, next_sources)
    elif reflection.follow_up_action == "web_search" and reflection.next_query:
        web_search_action = await web_search(app_state, reflection.next_query)
        next_sources = answer_sources.add_web_search(web_search_action)
        return (web_search_action, next_sources)
    elif reflection.follow_up_action == "stop":
        return (StopAction(), answer_sources)

async def reflection_single_step(app_state: AppState, message: str, message_history: Optional[List[ChatCompletionMessageParam]] = None) -> ActionList:
    question_action = QuestionAction(question_text=message)
    index_search_action = await index_search(app_state, message, message_history)
    answer_action = await generate_answer(app_state, message, AnswerSources.from_index_search(message_history, index_search_action))
    evaluation_action = await evaluate_answer(app_state, answer_action)
    action_list = ActionList(actions=[question_action, index_search_action, answer_action, evaluation_action ])
    if any(evaluation.score < 4 for evaluation in evaluation_action.evaluations):
        reflection_action = await reflect_answer(app_state, action_list)
        next_action, next_sources = await reflect_next_action(app_state, reflection_action, answer_action.answer_sources)
        final_answer_action = await generate_answer(app_state, message, next_sources, previous_answer=answer_action.answer_text, previous_answer_reflection=reflection_action.answer_reflection, previous_answer_evaluations=evaluation_action.evaluations)
        final_evaluation_action = await evaluate_answer(app_state, final_answer_action)

        action_list.actions.extend([reflection_action, next_action, final_answer_action, final_evaluation_action])
    
    return action_list

async def reflection_multi_step(app_state: AppState, message: str, message_history: Optional[List[ChatCompletionMessageParam]]) -> ActionList:
    question_action = QuestionAction(question_text=message)
    index_search_action = await index_search(app_state, message, message_history)
    answer_action = await generate_answer(app_state, message, AnswerSources.from_index_search(message_history, index_search_action))
    evaluation_action = await evaluate_answer(app_state, answer_action)
    action_list = ActionList(actions=[question_action, index_search_action, answer_action, evaluation_action ])
    max_turns = 3
    turn = 1
    while any(evaluation.score < 4 for evaluation in evaluation_action.evaluations):
        reflection_action = await reflect_answer(app_state, action_list)
        next_action, next_sources = await reflect_next_action(app_state, reflection_action, answer_action.answer_sources)
        next_answer_action = await generate_answer(app_state, message, next_sources, previous_answer=answer_action.answer_text, previous_answer_reflection=reflection_action.answer_reflection, previous_answer_evaluations=evaluation_action.evaluations)
        evaluation_action = await evaluate_answer(app_state, next_answer_action, reflection_action)
        action_list.actions.extend([reflection_action, next_action, next_answer_action, evaluation_action])

        if reflection_action.type == "stop":
            break
        turn += 1
        if turn > max_turns:
            break
    
    return action_list

