import os
from google import genai
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiMolecularInsights:
    """Generate AI-powered insights for drug molecules using Gemini API."""
    
    def __init__(self):
        """Initialize Gemini client with API key from environment."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.0-flash-exp"
    
    def generate_molecular_insights(
        self, 
        smiles: str,
        protein_target: str,
        binding_pkd: float,
        phase1_score: float,
        confidence: float,
        kd_nm: float,
        delta_g: float
    ) -> Dict[str, str]:
        """Generate comprehensive insights for a molecule."""
        
        prompt = f"""You are an expert medicinal chemist. Analyze this drug molecule and provide insights:

**Molecule Data:**
- SMILES: {smiles}
- Target Protein: {protein_target[:100]}
- Binding Affinity (pKd): {binding_pkd:.2f} (Kd: {kd_nm:.1f} nM)
- Free Energy (ΔG): {delta_g:.2f} kcal/mol
- Phase 1 Readiness: {phase1_score:.2f}
- Confidence: {confidence:.2f}

**Provide these insights:**

1. **SUMMARY** (2-3 sentences): What is this molecule? Overall assessment as a drug candidate.

2. **MEDICAL APPLICATIONS** (3-4 bullet points): 
   - What diseases/conditions could this treat?
   - What therapeutic areas is it relevant for?
   - Any similar approved drugs in this class?

3. **MECHANISM OF ACTION** (2-3 sentences): How does this molecule work? What biological processes does it affect?

4. **KEY MOLECULAR PROPERTIES** (3-4 bullet points):
   - Drug-likeness assessment
   - Bioavailability predictions
   - Safety/toxicity considerations
   - Selectivity and specificity

5. **DEVELOPMENT INSIGHTS** (2-3 sentences):
   - Is this a promising lead compound?
   - What stage of development would it be suitable for?
   - Key next steps for optimization

Keep responses clear, concise, and focused on practical drug discovery insights."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            text = response.text
            insights = self._parse_insights(text)
            return insights
            
        except Exception as e:
            return {
                "error": str(e),
                "summary": "Failed to generate insights",
                "medical_applications": "N/A",
                "mechanism": "N/A",
                "key_properties": "N/A",
                "development_insights": "N/A"
            }
    
    def _parse_insights(self, text: str) -> Dict[str, str]:
        """Parse Gemini response into structured sections."""
        sections = {
            "summary": "",
            "medical_applications": "",
            "mechanism": "",
            "key_properties": "",
            "development_insights": ""
        }
        
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if 'summary' in line_lower and ('**' in line or '#' in line):
                current_section = 'summary'
            elif 'medical' in line_lower or 'application' in line_lower and ('**' in line or '#' in line):
                current_section = 'medical_applications'
            elif 'mechanism' in line_lower and ('**' in line or '#' in line):
                current_section = 'mechanism'
            elif 'properties' in line_lower and ('**' in line or '#' in line):
                current_section = 'key_properties'
            elif 'development' in line_lower and ('**' in line or '#' in line):
                current_section = 'development_insights'
            elif current_section and line.strip() and not line.strip().startswith('#') and not line.strip().startswith('**'):
                sections[current_section] += line + "\n"
        
        for key in sections:
            sections[key] = sections[key].strip()
            if not sections[key]:
                sections[key] = "No insights generated for this section."
        
        return sections


def display_gemini_insights(insights: Dict, smiles: str, rank: int):
    """Display AI-generated insights (not used in new layout, kept for compatibility)."""
    pass
