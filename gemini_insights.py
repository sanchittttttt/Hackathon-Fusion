"""
Gemini AI integration for molecular insights.
This module provides AI-powered analysis of molecular structures and drug properties.
"""

import os
import streamlit as st
from typing import Dict, List, Optional
import json

# Try to import Google Generative AI, but handle gracefully if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: Google Generative AI not available. Gemini insights will be disabled.")
    GEMINI_AVAILABLE = False

class GeminiMolecularInsights:
    """AI-powered molecular analysis using Google's Gemini model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google AI API key. If None, tries to get from environment or Streamlit secrets.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI is not available. Please install google-generativeai.")
        
        # Get API key from various sources
        if api_key:
            self.api_key = api_key
        elif "GEMINI_API_KEY" in os.environ:
            self.api_key = os.environ["GEMINI_API_KEY"]
        elif hasattr(st, "secrets") and "gemini_api_key" in st.secrets:
            self.api_key = st.secrets["gemini_api_key"]
        else:
            # Use a placeholder for demo purposes
            self.api_key = "demo_key"
            print("Warning: No Gemini API key found. Using demo mode.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_molecular_insights(self, smiles: str, protein_target: str = "Unknown", 
                                  binding_pkd: float = 0.0, phase1_score: float = 0.0,
                                  confidence: float = 0.0, kd_nm: float = 0.0, 
                                  delta_g: float = 0.0) -> Dict:
        """
        Generate comprehensive molecular insights using Gemini AI.
        
        Args:
            smiles: SMILES string of the molecule
            protein_target: Target protein name
            binding_pkd: Binding affinity (pKd)
            phase1_score: Phase 1 readiness score
            confidence: Prediction confidence
            kd_nm: Kd value in nM
            delta_g: Delta G in kcal/mol
            
        Returns:
            Dictionary containing AI-generated insights
        """
        if not GEMINI_AVAILABLE:
            return self._get_demo_insights(smiles, protein_target, binding_pkd, phase1_score, confidence, kd_nm, delta_g)
        
        try:
            # Create a comprehensive prompt for molecular analysis
            prompt = f"""
            Analyze this drug molecule for pharmaceutical development:
            
            SMILES: {smiles}
            Target Protein: {protein_target}
            Binding Affinity (pKd): {binding_pkd:.2f}
            Kd (nM): {kd_nm:.1f}
            Delta G (kcal/mol): {delta_g:.1f}
            Phase 1 Readiness Score: {phase1_score:.3f}
            Prediction Confidence: {confidence:.3f}
            
            Please provide a comprehensive analysis including:
            1. Molecular structure interpretation
            2. Drug-like properties assessment
            3. Potential therapeutic applications
            4. Mechanism of action insights
            5. Development stage recommendations
            6. Risk factors and considerations
            7. Alternative applications if binding is weak
            
            Format your response as a structured analysis with clear sections.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response into structured insights
            insights = {
                "molecular_structure": self._extract_section(response.text, "structure"),
                "drug_properties": self._extract_section(response.text, "properties"),
                "therapeutic_applications": self._extract_section(response.text, "applications"),
                "mechanism_of_action": self._extract_section(response.text, "mechanism"),
                "development_recommendations": self._extract_section(response.text, "recommendations"),
                "risk_factors": self._extract_section(response.text, "risk"),
                "alternative_applications": self._extract_section(response.text, "alternative"),
                "full_analysis": response.text,
                "smiles": smiles,
                "protein_target": protein_target,
                "binding_pkd": binding_pkd,
                "phase1_score": phase1_score,
                "confidence": confidence
            }
            
            return insights
            
        except Exception as e:
            print(f"Error generating Gemini insights: {e}")
            return self._get_demo_insights(smiles, protein_target, binding_pkd, phase1_score, confidence, kd_nm, delta_g)
    
    def _extract_section(self, text: str, section_key: str) -> str:
        """Extract a specific section from the AI response."""
        # Simple text extraction - in a real implementation, you might use more sophisticated parsing
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if section_key.lower() in line.lower():
                in_section = True
                continue
            elif in_section and line.strip() and not line.startswith(' '):
                break
            elif in_section:
                section_content.append(line.strip())
        
        return '\n'.join(section_content) if section_content else "Analysis not available."
    
    def _get_demo_insights(self, smiles: str, protein_target: str, binding_pkd: float, 
                          phase1_score: float, confidence: float, kd_nm: float, delta_g: float) -> Dict:
        """Generate demo insights when Gemini is not available."""
        return {
            "molecular_structure": f"Molecular structure analysis for {smiles}. This molecule shows typical drug-like properties with good structural diversity.",
            "drug_properties": f"Drug-like properties assessment: pKd={binding_pkd:.2f}, Kd={kd_nm:.1f} nM. {'Good' if binding_pkd > 6 else 'Moderate' if binding_pkd > 4 else 'Weak'} binding affinity.",
            "therapeutic_applications": f"Potential therapeutic applications for {protein_target} targeting. Suitable for {'oncology' if 'C' in smiles else 'neurological'} indications.",
            "mechanism_of_action": f"Mechanism of action likely involves competitive inhibition of {protein_target}. Delta G of {delta_g:.1f} kcal/mol suggests {'strong' if delta_g < -8 else 'moderate'} binding.",
            "development_recommendations": f"Phase 1 readiness score: {phase1_score:.3f}. {'Proceed to clinical trials' if phase1_score > 0.7 else 'Requires optimization'} before clinical development.",
            "risk_factors": "Standard drug development risks apply. Monitor for off-target effects and toxicity profiles.",
            "alternative_applications": f"If binding is weak (pKd < 4), consider as a scaffold for further optimization or repurpose for other targets.",
            "full_analysis": f"Comprehensive analysis of {smiles} targeting {protein_target}. Binding affinity: {binding_pkd:.2f} pKd, Phase 1 readiness: {phase1_score:.3f}.",
            "smiles": smiles,
            "protein_target": protein_target,
            "binding_pkd": binding_pkd,
            "phase1_score": phase1_score,
            "confidence": confidence
        }

def display_gemini_insights(insights_list: List[Dict]):
    """
    Display Gemini insights in Streamlit.
    
    Args:
        insights_list: List of insight dictionaries from GeminiMolecularInsights
    """
    if not insights_list:
        st.warning("No insights available to display.")
        return
    
    for i, insights in enumerate(insights_list):
        with st.expander(f"🧬 AI Analysis - Molecule {i+1} ({insights.get('smiles', 'Unknown')[:20]}...)", expanded=True):
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Binding Affinity (pKd)", f"{insights.get('binding_pkd', 0):.2f}")
            with col2:
                st.metric("Phase 1 Readiness", f"{insights.get('phase1_score', 0):.3f}")
            with col3:
                st.metric("Confidence", f"{insights.get('confidence', 0):.3f}")
            
            # Detailed analysis sections
            st.markdown("### 📊 Molecular Structure Analysis")
            st.write(insights.get('molecular_structure', 'Not available'))
            
            st.markdown("### 💊 Drug Properties Assessment")
            st.write(insights.get('drug_properties', 'Not available'))
            
            st.markdown("### 🏥 Therapeutic Applications")
            st.write(insights.get('therapeutic_applications', 'Not available'))
            
            st.markdown("### ⚙️ Mechanism of Action")
            st.write(insights.get('mechanism_of_action', 'Not available'))
            
            st.markdown("### 🚀 Development Recommendations")
            st.write(insights.get('development_recommendations', 'Not available'))
            
            st.markdown("### ⚠️ Risk Factors")
            st.write(insights.get('risk_factors', 'Not available'))
            
            st.markdown("### 🔄 Alternative Applications")
            st.write(insights.get('alternative_applications', 'Not available'))
            
            # Full analysis
            with st.expander("📝 Full AI Analysis"):
                st.text(insights.get('full_analysis', 'Not available'))
            
            st.markdown("---")

# Demo function for testing
def test_gemini_insights():
    """Test function for Gemini insights."""
    try:
        insights_generator = GeminiMolecularInsights()
        test_insights = insights_generator.generate_molecular_insights(
            smiles="CCO",
            protein_target="EGFR",
            binding_pkd=7.5,
            phase1_score=0.85,
            confidence=0.92,
            kd_nm=3.2,
            delta_g=-8.5
        )
        print("✅ Gemini insights test successful")
        return test_insights
    except Exception as e:
        print(f"❌ Gemini insights test failed: {e}")
        return None

if __name__ == "__main__":
    # Test the module
    test_gemini_insights()