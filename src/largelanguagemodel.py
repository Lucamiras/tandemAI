# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# Environment
from dotenv import load_dotenv


load_dotenv()

class LargeLanguageModel:
    """
    A class used to represent a Large Language Model.
    Attributes
    ----------
    model : str or None
        The name of the model to use. If None, defaults to "gpt-3.5-turbo".
    temperature : float or None
        The temperature setting for the model, which controls the randomness of the output. If None, defaults to 0.9.
    llm : ChatOpenAI
        An instance of the ChatOpenAI model initialized with the specified parameters.
    Methods
    -------
    __init__(model: str|None=None, temperature: float|None=0.9)
        Initializes the LargeLanguageModel with the specified model and temperature.
    init_model()
        Initializes and returns an instance of the ChatOpenAI model with the specified parameters.
    """

    def __init__(self, temperature:float | None=0.9):
        self.temperature = temperature
        self.llm = self.init_model()

    def init_model(self):
        llm = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature=self.temperature or 0.7,
            max_tokens=1024,
        )
        return llm
    
        
class TandemPartner(LargeLanguageModel):
    """
    TandemPartner is a subclass of LargeLanguageModel designed to simulate a language learning partner.
    Methods
    -------
    __init__():
        Initializes the TandemPartner instance.
    generate_response(language, level, chat_input, name, chat_history):
        Generates a response based on the provided language, level, chat input, user name, and chat history.
        Parameters:
        language (str): The language the user is learning.
        level (str): The proficiency level of the user in the specified language.
        chat_input (str): The latest input from the user.
        name (str): The name of the user.
        chat_history (list): A list of tuples representing the chat history.
        Returns:
        str: The generated response from the TandemPartner.
    """
    
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
    """
    Critic class that extends the LargeLanguageModel class to provide functionality for generating responses
    that critique and correct language usage based on a given prompt.
    Methods
    -------
    __init__():
        Initializes the Critic class by calling the parent class's initializer.
    generate_response(language, source_language, level, chat_input):
        Generates a response that critiques and corrects the given chat input based on the specified language,
        source language, and proficiency level. The response follows a predefined JSON schema indicating whether
        there is a mistake and providing the corrected message if applicable.
        Parameters:
        - language (str): The target language for the critique.
        - source_language (str): The source language of the student.
        - level (str): The proficiency level of the student in the target language.
        - chat_input (str): The message input from the student to be critiqued.
        Returns:
        - response (dict): A dictionary containing the critique results following the JSON schema.
    """
    
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
    """
    A class used to represent a Translator that utilizes a large language model to generate translations.
    Methods
    -------
    __init__():
        Initializes the Translator instance.
    generate_response(language, source_language, word):
        Generates a translation for a given word from the source language to the target language.
        Parameters:
            language (str): The target language for the translation.
            source_language (str): The source language of the word to be translated.
            word (str): The word or phrase to be translated.
        Returns:
            dict: A dictionary containing the original word and its translation following the specified JSON schema.
    """

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