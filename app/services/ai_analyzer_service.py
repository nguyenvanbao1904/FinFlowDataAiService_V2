import logging
from langchain.chat_models import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIAnalyzerService:
    """Service responsible for AI-driven portfolio analysis."""
    
    def __init__(self):
        # Initialize LLM only if API key is provided
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY, 
                model_name="gpt-4-turbo-preview"
            )
        else:
            self.llm = None
            logger.warning("OPENAI_API_KEY not set. AI features will be mocked.")

    async def analyze_portfolio(self, portfolio_assets: list[dict], financial_indicators: list[dict]) -> str:
        """
        Analyzes a user's portfolio using LangChain and LLMs based on 
        the assets they hold and the corresponding financial indicators.
        """
        logger.info("Starting AI portfolio analysis...")
        
        if not self.llm:
            return "Mock AI Analysis: Danh mục của bạn có tỷ trọng công nghệ cao, ROE trung bình tốt (18.5%). Rủi ro ở mức trung bình."
            
        # TODO: Implement actual LangChain prompt chain
        # prompt = PromptTemplate.from_template("Analyze this portfolio: {assets} with indicators {indicators}")
        # chain = LLMChain(llm=self.llm, prompt=prompt)
        # result = await chain.arun(assets=portfolio_assets, indicators=financial_indicators)
        # return result
        
        return "AI analysis completed."
