import importlib.resources as pkg_resources

import json
import pickle
import pandas as pd
import openai
import google.generativeai as genai
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class RAGEvaluator:

    def __init__(self, query_engine, api_key, eval_mode = "default", speed = 'normal', 
                 mode = 'rag', questions = None , ground_truth_df = None):

        self.query_engine = query_engine
        self.api_key = api_key

        if eval_mode == "default":
          with pkg_resources.open_binary('rag_evaluator.data', 'all_questions.pkl') as f:
            self.questions = pickle.load(f)

          # Load the CSV file
          with pkg_resources.open_text('rag_evaluator.data', 'ground_truth_dataset.csv') as f:
                self.retrieval_ground_truth = pd.read_csv(f)

        elif eval_mode == "custom":
          if questions is None or ground_truth_df is None :
            raise ValueError("If eval_mode is 'custom', both questions and ground_truth_df must be provided.")
          self.questions = questions
          self.retrieval_ground_truth = ground_truth_df

        if speed != 'normal':
            self.questions = self.questions[:10]
            self.retrieval_ground_truth = self.retrieval_ground_truth.head(10)

        print('Initializing the evaluator')
        self.responses = [self.answer_question(question , mode) for question in tqdm(self.questions)]
        self.answers = [response.response for response in self.responses if response]
        self.contexts = [[node.node.get_text() for node in response.source_nodes] for response in self.responses if response]
        self.embedding_model = 'BAAI/bge-small-en-v1.5'
        self.initialize_evaluation_model()
        self.initialize_embedding_model()
        print('Initialization complete')

    def initialize_evaluation_model(self):
        self.evaluation_model = openai.OpenAI(
                    api_key= self.api_key,
                    base_url="https://glhf.chat/api/openai/v1",)
        
    def initialize_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.embedding_model)

    def answer_question(self, question: str, mode = 'rag'):
        try:
            if mode == 'rag' :
                result = self.query_engine.query(question)
            elif mode == 'agentic':
                self.query_engine.reset()
                result = self.query_engine.chat(question)
            return result
        
        except Exception as e:
            print(f"Error answering question: {e}")
            return None
    
    def generate_evaluation(self , prompt:str) -> str:
        try :
            completion = self.evaluation_model.chat.completions.create(
                                            model="hf:meta-llama/Llama-3.1-405B-Instruct",
                                            messages=[{"role": "user", "content": prompt}]
                                            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion : {e}")
            return None

    def check_correctness(self, questions: List, answers: List) -> float:
        correctness_prompt = """Evaluate the following answer in response to the question. Provide a score from 0 to 10 based on the following criteria:
        1. **Writing Correctness**: Is the answer clearly written, grammatically correct, and free from any spelling or syntactical errors?
        2. **Relevance**: Does the answer directly address the question, staying on-topic and providing relevant information without introducing irrelevant details?
        3. **Goal Alignment**: Does the answer meet the actual goal of the query and provide a comprehensive and informative response that fulfills the user's intent?
        4. **Clarity and Precision**: Is the answer clear, precise, and easy to understand, without being overly verbose or ambiguous?

        Please return your evaluation as a dictionary with two keys:
        - **'score'**: The numerical score out of 10.
        - **'explanation'**: A detailed explanation of the score, describing the reasoning behind the evaluation. Without any other text.

        Question: {question}
        Answer: {answer}"""

        def evaluate_single(q, a):
            try:
                result = self.generate_evaluation(correctness_prompt.format(question=q, answer=a))
                print(result)
                result_text = re.sub("```json|```", "", result)
                result_dict = json.loads(result_text)
                return result_dict['score']
            except Exception as e:
                print(f"Error evaluating correctness: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            scores = list(tqdm(executor.map(evaluate_single, questions, answers), total=len(questions)))

        scores = [score for score in scores if score is not None]
        return np.mean(scores)

    def check_faithfulness(self, contexts: List, answers: List) -> float:
        faithfulness_prompt = """You are given the following information retrieved for a query. Evaluate how well the following answer reflects the insights retrieved. Specifically, check whether the answer accurately uses the information from the retrieval, without introducing any distortions, omissions, or misrepresentations. The answer should stay true to the content of the retrieved information.

        Please evaluate the answer against the retrieved insights, considering the following criteria:

        1. **Accuracy**: Does the answer faithfully reflect the information retrieved, without introducing incorrect details or omitting important facts?
        2. **Consistency**: Does the answer maintain the same meaning as the retrieved information, avoiding rewording that changes the original intent?
        3. **Integrity**: Is the answer grounded in the retrieved information and free from misinterpretations or distortions?

        Answer: {answer}
        Retrieved Insight: {retrieved_insight}

        Please return your evaluation as a dictionary with two keys:
        - **'score'**: The numerical score out of 10, reflecting how well the answer uses the retrieved information.
        - **'explanation'**: A detailed explanation of the score, describing the reasoning behind the evaluation and how well the answer adhered to the retrieved data.
        Without any other text."""

        def evaluate_single(c, a):
            try:
                retrieved_insight = " ".join(c)  # Concatenate all context chunks
                result = self.generate_evaluation(
                    faithfulness_prompt.format(answer=a, retrieved_insight=retrieved_insight)
                )
                result_text = re.sub("```json|```", "", result)
                result_dict = json.loads(result_text)
                return result_dict['score']
            except Exception as e:
                print(f"Error evaluating faithfulness: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            scores = list(tqdm(executor.map(evaluate_single, contexts, answers), total=len(contexts)))

        scores = [score for score in scores if score is not None]
        return np.mean(scores)

    def check_hallucinations(self, questions: List, answers: List) -> float:
        hallucination_prompt = """You are given the following question. Your task is to rewrite this question in three different ways while preserving the original intent.
        The rephrased questions should maintain the meaning and scope of the original question, ensuring they can still be answered with the same information.

        Question: {question}

        Please return three rewritten versions of this question, making sure that:
        1. The original meaning is fully preserved.
        2. The reworded versions are distinct, providing variety in phrasing without changing the original intent.

        Format the output with numbered versions:
        1. Version 1
        2. Version 2
        3. Version 3
        Without any other text"""

        similarity_scores = []

        for question, original_answer in tqdm(zip(questions, answers), total=len(questions)):
            try:
                rewrited_questions = self.generate_evaluation(
                    hallucination_prompt.format(question=question)
                )
                rewrited_questions = [q.strip() for q in rewrited_questions.split('\n') if q.strip()]
                rewrited_questions = [
                    q[q.find(".") + 1 :].strip() if q[0].isdigit() else q for q in rewrited_questions
                ]  # Strip numbering

                answers = [self.answer_question(q) for q in rewrited_questions if q]
                answers = [answer.response for answer in answers]

                answer_embeddings = self.embedding_model.encode([original_answer] + answers)
                for i in range(1, len(answer_embeddings)):
                    sim = cosine_similarity([answer_embeddings[0]], [answer_embeddings[i]])[0][0]
                    score = np.interp(sim, (0.6, 1), (1, 10)) if sim >= 0.6 else 1  
                    similarity_scores.append(score)
            except Exception as e:
                print(f"Error evaluating hallucinations: {e}")
                continue

        return np.mean(similarity_scores)

    def evaluate_retrieval(self, questions: List, contexts:List, ground_truth_df: pd.DataFrame, k: int = 5) -> float:
        all_scores = []

        for question , context in tqdm(zip(questions , contexts), total=len(questions)):

          try:
            # Retrieve chunks for the question
            retrieved_chunks = context

            # Get ground truth (relevant chunks)
            ground_truth = ground_truth_df.loc[ground_truth_df['question'] == question, 'ranked_chunks']
            print(len(ground_truth))
            print(ground_truth.head())

            # Create a binary relevance list (1 for relevant, 0 for non-relevant)
            relevance = [1 if doc in ground_truth else 0 for doc in retrieved_chunks[:k]]

            # Calculate Average Precision for the current question
            avg_precision = average_precision_score([1] * len(ground_truth), relevance)

            # Scale the score from 0-1 to 0-10
            scaled_score = np.interp(avg_precision, (0, 1), (0, 10))
            all_scores.append(scaled_score)

          except Exception as e:
            print(f"Exception: {e}")
            continue

        return np.mean(all_scores)


    def evaluate_rag(self):
        print('Beginning the evaluation')
        print('Evaluating Correctness')
        correctness_score = self.check_correctness(self.questions, self.answers)
        print('Evaluating Faithfulness')
        faithfulness_score = self.check_faithfulness(self.contexts, self.answers)
        print('Evaluating Retrieval')
        retrieval_score = self.evaluate_retrieval(self.questions, self.contexts, self.retrieval_ground_truth)
        print('Evaluating Hallucination')
        hallucination_score = self.check_hallucinations(self.questions, self.answers)
        print('Evaluation complete')

        return {
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            "retrieval_score": retrieval_score,
            "hallucination_score": hallucination_score,
            "total_score": (correctness_score + faithfulness_score + retrieval_score + hallucination_score) / 4,
        }
