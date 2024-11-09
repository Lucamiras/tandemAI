# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# Environment
from dotenv import load_dotenv


load_dotenv()

class LargeLanguageModel:

    def __init__(self, model:str|None=None, temperature:float|None=0.9):
        self.model = model
        self.temperature = temperature
        self.llm = self.init_model()

        
    def init_model(self):
        llm = ChatOpenAI(
            model = self.model or "gpt-3.5-turbo",
            temperature=self.temperature or 0.7,
            max_tokens=1024,
        )
        return llm
    
        
class TandemPartner(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        
    def generate_response(self, language, level, chat_input, name, chat_history):
        system_prompt = """You are a {language} native speaker, teacher and TANDEM PARTNER. The person talking to you is called {user_name}, an aspiring {level} {language} student. 
                        For anything they write in {language}, you should respond in {language}. Consider their level.
                        If the student is asking you a question, answer it in the style of a friend chatting via text.
                        If the student is not asking you a question, ask them about their day, their interests, or anything else you'd like to know.
                        Always stay friendly and positive!
                        Keep your responses concise. Keep the conversation going. DO NOT ask them if you can help them!"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            *chat_history,
            ("human", "{chat_input}")
        ])

        # prompt = ChatPromptTemplate.from_messages([
        #    ("system", """You are a {language} native speaker, teacher and TANDEM PARTNER. The person talking to you is called {user_name}, an aspiring {level} {language} student. 
        #                For anything they write in {language}, you should respond in {language}. Consider their level.
        #                If the student is not asking you a question, ask them about their day, their interests, or anything else you'd like to know.
        #                Keep your responses concise. Keep the conversation going. DO NOT ask them if you can help them!"""),
        #    ("human", "{chat_input}")]
        # )

        chain = template | self.llm | StrOutputParser()
        response = chain.invoke(
            {
                "language": language,
                "level": level,
                "chat_input": chat_input,
                "user_name": name
            }
        )
        return response
    

class Critic(LargeLanguageModel):
    def __init__(self):
        super().__init__()

    def generate_response(self, language, source_language, level, chat_input):
        json_schema = """{{
            "mistake_boolean": true,
            "original_message": "Ich habe ein Frage.",
            "correction": "It should be 'Ich habe einE Frage.'"
        }}"""
        
        prompt = """
        You are a {language} native speaker and a CRITIC. 
        You are observing a conversation between a {language} native speaker and a {level} {language} student.
        You will only see the student's messages. If you see a mistake, correct it.
        Follow this json schema: {json_schema}. The key 'mistake_boolean' should be true if there is a mistake. Else it should be false.
        Here is the message: {user_message}"""
        
        structured_llm = self.llm.with_structured_output(json_schema, method="json_mode")
        prompt_template = PromptTemplate(
            input_variables=['language', 'level', 'user_message', 'json_schema'],
            template=prompt
        )
        chain = prompt_template | structured_llm
        response = chain.invoke({
            "language": language,
            "source_language": source_language,
            "level": level,
            "user_message": chat_input,
            "json_schema": json_schema
        })
        return response


class Translator(LargeLanguageModel):
    def __init__(self):
        super().__init__()

    def generate_response(self, language, source_language, word):
        json_schema = """{{
            "word": "Uebersetzung",
            "translation": "translation",
        }}"""
        
        prompt = """
        You are a {language} native speaker and a translator. 
        At the end of this prompt there is a word or phrase. Translate it to {source_language}.
        Use the root form of the word if possible. If the word is a verb, use the infinitive form.
        If you don't know the word, translate it as UNKNOWN.
        Follow this json schema: {json_schema}. The key 'word' should be the word and the key 'translation' should be the translation.
        Here is the word: {word}"""
        
        structured_llm = self.llm.with_structured_output(json_schema, method="json_mode")
        prompt_template = PromptTemplate(
            input_variables=['language', 'source_language', 'word', 'json_schema'],
            template=prompt
        )
        chain = prompt_template | structured_llm
        response = chain.invoke({
            "language": language,
            "source_language": source_language,
            "word": word,
            "json_schema": json_schema
        })
        return response

if __name__ == "__main__":
    print("This is a helper file for langchain functions")