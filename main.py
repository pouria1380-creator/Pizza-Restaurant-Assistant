from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import ReviewRetriever

class PizzaRestaurantAssistant:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2")
        self.retriever = ReviewRetriever()
        self._initialize_prompt_template()
        
    def _initialize_prompt_template(self):
        template = """
        you are an expert in answering questions about a pizza restaurant
        here are some relevant reviews: {reviews}
        here is the question to answer: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model
    
    def get_response(self, question):
        reviews = self.retriever.invoke(question)
        response = self.chain.invoke(
            {"reviews": reviews, "question": question}
        )
        return response
    
    def run(self):
        while True:
            question = input("What do you want to know? (type 'exit' to end): ")
            
            if question.lower() == 'exit':
                break
                
            print('\nThe answer:')
            response = self.get_response(question)
            print(response + '\n')

if __name__ == "__main__":
    assistant = PizzaRestaurantAssistant()
    assistant.run()