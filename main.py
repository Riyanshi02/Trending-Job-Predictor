import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-prob { border-left: 5px solid #22c55e; }
    .medium-prob { border-left: 5px solid #f59e0b; }
    .low-prob { border-left: 5px solid #ef4444; }
    .requirements-missing {
        background: #fef3c7;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #f59e0b;
    }
    .requirements-met {
        background: #dcfce7;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #22c55e;
    }
    .speedometer-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

class PlacementPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.is_trained = False
        self.model_files_exist = False
    
    @st.cache_data
    def load_datasets(_self):
        """Load all datasets with caching"""
        try:
            _self.data = {
                'companies': pd.read_csv('companies dataset.csv'),
                'skills': pd.read_csv('skills master.csv'),
                'placements': pd.read_csv('historical placement.csv'),
                'company_skills': pd.read_csv('company skill.csv'),
                'certifications': pd.read_csv('certification.csv'),
                'skill_development': pd.read_csv('skill development.csv'),
                'project_impact': pd.read_csv('project impact.csv'),
                'internship_impact': pd.read_csv('internship impact.csv'),
                'placement_prob': pd.read_csv('placement probability.csv'),
                'branch_company': pd.read_csv('branch-wise company.csv'),
                'sample': pd.read_csv('sample dataset.csv')
            }
            return True
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
            return False
    
    def check_model_files(self):
        """Check if model files exist"""
        required_files = [
            'random_forest_model.pkl',
            'logistic_regression_model.pkl',
            'gradient_boosting_model.pkl',
            'placement_scaler.pkl'
        ]
        
        self.model_files_exist = all(os.path.exists(f) for f in required_files)
        return self.model_files_exist
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        if not self.check_model_files():
            return False
            
        try:
            # Load models
            with open('random_forest_model.pkl', 'rb') as f:
                self.models['random_forest'] = pickle.load(f)
            
            with open('logistic_regression_model.pkl', 'rb') as f:
                self.models['logistic_regression'] = pickle.load(f)
            
            with open('gradient_boosting_model.pkl', 'rb') as f:
                self.models['gradient_boosting'] = pickle.load(f)
            
            # Load scaler
            with open('placement_scaler.pkl', 'rb') as f:
                self.scalers['placement'] = pickle.load(f)
            
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def parse_skills_string(self, skills_str):
        """Parse skills from string format"""
        if pd.isna(skills_str) or skills_str == '':
            return []
        try:
            if isinstance(skills_str, str):
                skills_str = skills_str.strip('"').replace('"', '')
                return [int(x.strip()) for x in skills_str.split(',') if x.strip().isdigit()]
            return []
        except:
            return []
    
    def calculate_skill_match_score(self, student_skills, company_id):
        """Calculate comprehensive skill match score"""
        company_reqs = self.data['company_skills'][
            self.data['company_skills']['company_id'] == company_id
        ]
        
        if len(company_reqs) == 0:
            return 0.5, [], []
        
        mandatory_skills = company_reqs[
            company_reqs['requirement_type'] == 'Mandatory'
        ]['skill_id'].tolist()
        
        preferred_skills = company_reqs[
            company_reqs['requirement_type'] == 'Preferred'
        ]['skill_id'].tolist()
        
        # Find missing skills
        missing_mandatory = list(set(mandatory_skills) - set(student_skills))
        missing_preferred = list(set(preferred_skills) - set(student_skills))
        
        # Calculate match scores
        mandatory_match = 0
        if mandatory_skills:
            mandatory_match = len(set(student_skills) & set(mandatory_skills)) / len(mandatory_skills)
        else:
            mandatory_match = 1.0  # If no mandatory skills, consider it fully matched
        
        preferred_match = 0
        if preferred_skills:
            preferred_match = len(set(student_skills) & set(preferred_skills)) / len(preferred_skills)
        
        # Weighted final score
        final_score = (mandatory_match * 0.8) + (preferred_match * 0.2)
        
        return min(final_score, 1.0), missing_mandatory, missing_preferred
    
    def get_branch_preference_score(self, branch, company_id):
        """Get branch preference score"""
        try:
            branch_row = self.data['branch_company'][
                self.data['branch_company']['company_id'] == company_id
            ]
            
            if len(branch_row) == 0:
                return 0.7  # Default neutral score
            
            branch_col = f"{branch.lower()}_preference"
            if branch_col in branch_row.columns:
                return float(branch_row[branch_col].iloc[0])
            return 0.7
        except:
            return 0.7
    
    def calculate_certification_boost(self, certifications, company_tier):
        """Calculate certification boost"""
        if not certifications:
            return 0.0
        
        total_boost = 0.0
        cert_list = [cert.strip() for cert in certifications.split(',') if cert.strip()]
        
        for cert in cert_list:
            matching_certs = self.data['certifications'][
                self.data['certifications']['certification_name'].str.contains(
                    cert[:10], case=False, na=False
                )
            ]
            
            if len(matching_certs) > 0:
                cert_data = matching_certs.iloc[0]
                
                if company_tier == 'Tier1':
                    total_boost += cert_data.get('tier1_boost', 0)
                elif company_tier == 'Tier2':
                    total_boost += cert_data.get('tier2_boost', 0)
                else:
                    total_boost += cert_data.get('tier3_boost', 0)
        
        return min(total_boost, 0.4)  # Cap the boost
    
    def get_internship_impact(self, internship_count, company_tier):
        """Calculate internship impact score"""
        if internship_count == 0:
            return 0.0
        
        # Base impact increases with more internships but with diminishing returns
        base_impact = min(internship_count * 0.15, 0.5)
        
        # Tier multiplier
        tier_multiplier = {'Tier1': 1.2, 'Tier2': 1.1, 'Tier3': 1.0}
        multiplier = tier_multiplier.get(company_tier, 1.0)
        
        return min(base_impact * multiplier, 0.6)
    
    def get_project_impact(self, project_count, company_tier):
        """Calculate project impact score"""
        if project_count == 0:
            return 0.0
        
        # Projects have strong impact especially for technical roles
        base_impact = min(project_count * 0.12, 0.4)
        
        # Tier multiplier
        tier_multiplier = {'Tier1': 1.3, 'Tier2': 1.15, 'Tier3': 1.0}
        multiplier = tier_multiplier.get(company_tier, 1.0)
        
        return min(base_impact * multiplier, 0.5)
    
    def predict_company_probability(self, student_profile, company_id):
        """Enhanced prediction considering all factors"""
        try:
            company_info = self.data['companies'][
                self.data['companies']['company_id'] == company_id
            ]
            
            if len(company_info) == 0:
                return 0.0, {}
            
            company_info = company_info.iloc[0]
            
            cgpa = student_profile['cgpa']
            branch = student_profile['branch']
            skills = student_profile.get('skills', [])
            internship_count = student_profile.get('internship_count', 0)
            project_count = student_profile.get('project_count', 0)
            certifications = student_profile.get('certifications', '')
            
            # Calculate individual factors
            skill_match, missing_mandatory, missing_preferred = self.calculate_skill_match_score(skills, company_id)
            branch_pref = self.get_branch_preference_score(branch, company_id)
            cgpa_eligible = 1 if cgpa >= company_info['min_cgpa'] else 0
            cert_boost = self.calculate_certification_boost(certifications, company_info['tier'])
            internship_impact = self.get_internship_impact(internship_count, company_info['tier'])
            project_impact = self.get_project_impact(project_count, company_info['tier'])
            
            # Enhanced scoring system
            base_score = 0.0
            
            # CGPA factor (25% weight)
            if cgpa_eligible:
                cgpa_factor = min((cgpa - company_info['min_cgpa']) / 2.0 + 0.5, 1.0)
                base_score += cgpa_factor * 0.25
            else:
                # Heavy penalty for not meeting CGPA
                base_score -= 0.3
            
            # Skills factor (35% weight)
            base_score += skill_match * 0.35
            
            # Branch preference (15% weight)  
            base_score += branch_pref * 0.15
            
            # Experience factors (25% weight total)
            base_score += internship_impact * 0.15
            base_score += project_impact * 0.10
            
            # Certification boost (additional)
            base_score += cert_boost
            
            # Company selectivity adjustment
            selection_ratio = company_info['selection_ratio']
            selectivity_factor = min(selection_ratio * 2.0, 1.0)  # More selective = harder
            
            # Final probability
            final_prob = base_score * selectivity_factor
            
            # Ensure probability is within bounds
            final_prob = max(0.0, min(1.0, final_prob))
            
            # Detailed breakdown for analysis
            breakdown = {
                'cgpa_score': cgpa_factor * 0.25 if cgpa_eligible else -0.3,
                'skill_score': skill_match * 0.35,
                'branch_score': branch_pref * 0.15,
                'internship_score': internship_impact * 0.15,
                'project_score': project_impact * 0.10,
                'cert_score': cert_boost,
                'missing_mandatory_skills': missing_mandatory,
                'missing_preferred_skills': missing_preferred,
                'cgpa_requirement_met': cgpa_eligible,
                'min_cgpa_required': company_info['min_cgpa']
            }
            
            return final_prob, breakdown
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0, {}
    
    def get_skill_name(self, skill_id):
        """Get skill name from skill ID"""
        skill_info = self.data['skills'][self.data['skills']['skill_id'] == skill_id]
        if len(skill_info) > 0:
            return skill_info.iloc[0]['skill_name']
        return f"Skill {skill_id}"
    
    def get_skill_recommendations(self, student_profile, target_companies):
        """Generate comprehensive skill recommendations"""
        student_skills = set(student_profile.get('skills', []))
        
        required_skills = {}
        skill_importance = {}
        
        for company_id in target_companies:
            company_reqs = self.data['company_skills'][
                self.data['company_skills']['company_id'] == company_id
            ]
            
            company_tier = self.data['companies'][
                self.data['companies']['company_id'] == company_id
            ]['tier'].iloc[0]
            
            tier_weight = {'Tier1': 3, 'Tier2': 2, 'Tier3': 1}.get(company_tier, 1)
            
            for _, req in company_reqs.iterrows():
                skill_id = req['skill_id']
                req_weight = 3 if req['requirement_type'] == 'Mandatory' else 1
                
                if skill_id not in student_skills:
                    required_skills[skill_id] = True
                    skill_importance[skill_id] = skill_importance.get(skill_id, 0) + (tier_weight * req_weight)
        
        recommendations = []
        for skill_id in required_skills:
            skill_info = self.data['skills'][self.data['skills']['skill_id'] == skill_id]
            
            if len(skill_info) > 0:
                skill_info = skill_info.iloc[0]
                
                learning_path = self.data['skill_development'][
                    self.data['skill_development']['missing_skill_id'] == skill_id
                ]
                
                rec = {
                    'skill_id': int(skill_id),
                    'skill_name': skill_info['skill_name'],
                    'category': skill_info['category'],
                    'difficulty': skill_info['difficulty_level'],
                    'learning_time_months': int(skill_info['learning_time_months']),
                    'importance_score': skill_importance.get(skill_id, 0),
                    'market_demand': int(skill_info['market_demand_score']),
                    'learning_path': '',
                    'resources': '',
                    'priority': 'High' if skill_importance.get(skill_id, 0) > 6 else 'Medium' if skill_importance.get(skill_id, 0) > 3 else 'Low'
                }
                
                if len(learning_path) > 0:
                    path_info = learning_path.iloc[0]
                    rec['learning_path'] = path_info['learning_path']
                    rec['resources'] = path_info['resources']
                    rec['difficulty_boost'] = int(path_info.get('difficulty_boost', 0))
                
                recommendations.append(rec)
        
        recommendations.sort(
            key=lambda x: (x['importance_score'], x['market_demand']), 
            reverse=True
        )
        
        return recommendations[:12]

def create_speedometer(probability):
    """Create a speedometer gauge for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Trending Job Predictor", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgray'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# Initialize the predictor
@st.cache_resource
def initialize_predictor():
    """Initialize predictor with models and data"""
    predictor = PlacementPredictor()
    if predictor.load_datasets():
        predictor.load_models()  # This will set is_trained based on success
        return predictor
    return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Trending Job Predictor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown('<h1 class="main-header">Trending Job Predictor Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Career Guidance with Trending Job Predictor & Skill Recommendations</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = initialize_predictor()
    
    if predictor is None:
        st.error("❌ Failed to initialize predictor. Please ensure all dataset files are available.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Job Prediction", "📊 Analytics Dashboard", "🎯 Skill Recommendations"],
        index=["🏠 Home", "🔮 Job Prediction", "📊 Analytics Dashboard", "🎯 Skill Recommendations"].index(st.session_state.page) if st.session_state.page in ["🏠 Home", "🔮 Job Prediction", "📊 Analytics Dashboard", "🎯 Skill Recommendations"] else 0
    )
    
    st.session_state.page = page
    
    if page == "🏠 Home":
        home_page(predictor)
    elif page == "🔮 Job Prediction":
        prediction_page(predictor)
    elif page == "📊 Analytics Dashboard":
        analytics_page(predictor)
    elif page == "🎯 Skill Recommendations":
        recommendations_page(predictor)

def home_page(predictor):
    st.header("🏠 Welcome to Trending Job Predictor")
    
    # System stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        company_count = len(predictor.data['companies'])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{company_count}+</h2>
            <p>Companies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        skill_count = len(predictor.data['skills'])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{skill_count}+</h2>
            <p>Skills Tracked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        req_count = len(predictor.data['company_skills'])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{req_count}+</h2>
            <p>Job Requirements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        placement_count = len(predictor.data['placements'])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{placement_count}+</h2>
            <p>Historical Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features overview
    st.subheader("🚀 What You Can Do")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        ### 🔮 Job Prediction
        - Get personalized job probability for each company
        - Individual speedometers for every company
        - Detailed insights with company-wise analysis
        - Package predictions based on your profile""")

    with col2:
        st.error("""
        ### 📊 Market Analytics
        - Industry trends and insights
        - Company hiring patterns
        - Skills demand analysis
        - Placement success factors""")

    col1, col2 = st.columns(2) 
    with col1:
        st.warning("""
        ### 🎯 Skill Recommendations
        - Identify skills gap for target companies
        - Personalized learning roadmap
        - Priority-based skill development plan
        - Timeline and resource recommendations""")

    with col2:
        st.info("""
        ### 🎓 Career Guidance
        - CGPA impact analysis
        - Branch-wise opportunities
        - Certification value assessment
        - Strategic career planning""")
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("⚡ Quick Start Guide")
    
    st.markdown("""
    1. **🔮 Start with Job Prediction** - Enter your profile details and get instant predictions
    2. **🎯 Get Skill Recommendations** - Select target companies and discover missing skills
    3. **📊 Explore Analytics** - Understand market trends and job patterns
    4. **📈 Track Progress** - Regularly update your profile and monitor improvements
    """)
    
    # Call to action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Job Prediction", type="primary", use_container_width=True):
            st.session_state.page = "🔮 Job Prediction"
            st.rerun()

def prediction_page(predictor):
    st.header("🔮 Job Probability Prediction")
    
    # Student profile input
    st.subheader("👤 Your Profile")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.0, step=0.05)
        branch = st.selectbox("Branch", ["Computer Science Engineering", "Electronics and Communication Engineering", "Information Technology", "Electrical Engineering", "Mechanical Engineering"])
        internships = st.number_input("Internships", min_value=0, max_value=10, value=1)
        projects = st.number_input("Projects", min_value=0, max_value=20, value=1)
    
    with col2:
        certifications = st.text_area("Certifications (comma-separated)", 
                                    value="AWS Cloud Practitioner, Google Cloud")
        
        # Skills selection
        skills_options = predictor.data['skills'][['skill_id', 'skill_name', 'category']].to_dict('records')
        skill_names = [f"{skill['skill_name']} ({skill['category']})" for skill in skills_options]
        
        selected_skills = st.multiselect(
            "Select your skills:",
            skill_names,
            default=skill_names[:2]  # Default first 5 skills
        )
        
        # Convert selected skills back to IDs
        selected_skill_ids = []
        for selected in selected_skills:
            for skill in skills_options:
                if f"{skill['skill_name']} ({skill['category']})" == selected:
                    selected_skill_ids.append(skill['skill_id'])
                    break
    
    if st.button("🔮 Predict Placements", type="primary"):
        student_profile = {
            'cgpa': cgpa,
            'branch': branch,
            'skills': selected_skill_ids,
            'internship_count': internships,
            'project_count': projects,
            'certifications': certifications
        }
        
        # Generate predictions
        predictions = []
        for _, company in predictor.data['companies'].iterrows():
            prob, breakdown = predictor.predict_company_probability(student_profile, company['company_id'])
            predictions.append({
                'company_name': company['company_name'],
                'tier': company['tier'],
                'probability': prob * 100,
                'min_cgpa': company['min_cgpa'],
                'avg_package': company['avg_package_lpa'],
                'max_package': company['max_package_lpa'],
                'location': company['location'],
                'breakdown': breakdown,
                'company_id': company['company_id']
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display results
        st.subheader("🎯 Placement Predictions")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        high_prob = len([p for p in predictions if p['probability'] >= 70])
        medium_prob = len([p for p in predictions if 30 <= p['probability'] < 70])
        low_prob = len([p for p in predictions if p['probability'] < 30])
        avg_package = np.mean([p['avg_package'] for p in predictions if p['probability'] >= 50])
        
        with col1:
            st.metric("High Probability (≥70%)", high_prob)
        with col2:
            st.metric("Medium Probability (30-69%)", medium_prob)
        with col3:
            st.metric("Low Probability (<30%)", low_prob)
        with col4:
            st.metric("Avg Package (High Prob)", f"₹{avg_package:.1f}L")
        
        # Top prediction with speedometer
        if predictions:
            st.subheader("🏆 Top Opportunity")
            top_pred = predictions[0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                speedometer_fig = create_speedometer(top_pred['probability'] / 100)
                st.plotly_chart(speedometer_fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                ### {top_pred['company_name']} ({top_pred['tier']})
                **📍 Location:** {top_pred['location']}  
                **💰 Package:** ₹{top_pred['avg_package']:.1f}L - ₹{top_pred['max_package']:.1f}L  
                **📚 Min CGPA:** {top_pred['min_cgpa']}  
                **🎯 Your Match:** {top_pred['probability']:.1f}%
                """)
                
                # Requirements analysis
                breakdown = top_pred['breakdown']
                if breakdown.get('cgpa_requirement_met', True):
                    st.markdown('<div class="requirements-met">✅ CGPA Requirement Met</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="requirements-missing">❌ CGPA Required: {breakdown.get("min_cgpa_required", "N/A")}</div>', unsafe_allow_html=True)
                
                # Missing skills
                if breakdown.get('missing_mandatory_skills'):
                    missing_skills = [predictor.get_skill_name(sid) for sid in breakdown['missing_mandatory_skills']]
                    st.markdown(f'<div class="requirements-missing">🔧 Missing Mandatory Skills: {", ".join(missing_skills[:3])}</div>', unsafe_allow_html=True)
                
                if breakdown.get('missing_preferred_skills'):
                    missing_pref = [predictor.get_skill_name(sid) for sid in breakdown['missing_preferred_skills']]
                    st.markdown(f'<div class="requirements-missing">⭐ Missing Preferred Skills: {", ".join(missing_pref[:3])}</div>', unsafe_allow_html=True)
        
        # Top predictions visualization
        top_10 = predictions[:10]
        companies = [p['company_name'] for p in top_10]
        probabilities = [p['probability'] for p in top_10]
        
        fig = px.bar(x=probabilities, y=companies, orientation='h',
                    title="Top 10 Company Predictions",
                    labels={'x': 'Probability (%)', 'y': 'Company'},
                    color=probabilities,
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed predictions table with requirements
        st.subheader("📋 Detailed Predictions & Requirements")
        
        for i, pred in enumerate(predictions[:15]):
            prob_class = "high-prob" if pred['probability'] >= 70 else "medium-prob" if pred['probability'] >= 30 else "low-prob"
            
            breakdown = pred['breakdown']
            
            # Requirements status
            requirements_html = ""
            if breakdown.get('cgpa_requirement_met', True):
                requirements_html += '<span style="color: green;">✅ CGPA</span> '
            else:
                requirements_html += f'<span style="color: red;">❌ CGPA (Need {breakdown.get("min_cgpa_required", "N/A")})</span> '
            
            # Skill requirements
            missing_mandatory = breakdown.get('missing_mandatory_skills', [])
            missing_preferred = breakdown.get('missing_preferred_skills', [])
            
            if not missing_mandatory:
                requirements_html += '<span style="color: green;">✅ Mandatory Skills</span> '
            else:
                requirements_html += f'<span style="color: red;">❌ Missing {len(missing_mandatory)} Mandatory Skills</span> '
            
            if not missing_preferred:
                requirements_html += '<span style="color: green;">✅ Preferred Skills</span>'
            else:
                requirements_html += f'<span style="color: orange;">⚠️ Missing {len(missing_preferred)} Preferred Skills</span>'
            
            # Score breakdown
            score_breakdown = f"""
            <small>
            📊 Score Breakdown: 
            CGPA: {breakdown.get('cgpa_score', 0):.2f} | 
            Skills: {breakdown.get('skill_score', 0):.2f} | 
            Branch: {breakdown.get('branch_score', 0):.2f} | 
            Experience: {breakdown.get('internship_score', 0) + breakdown.get('project_score', 0):.2f} |
            Certs: {breakdown.get('cert_score', 0):.2f}
            </small>
            """
            
            st.markdown(f"""
            <div class="prediction-card {prob_class}">
                <h4>{pred['company_name']} ({pred['tier']})</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div>
                        <strong>Probability: {pred['probability']:.1f}%</strong><br>
                        Location: {pred['location']}<br>
                        Min CGPA: {pred['min_cgpa']}
                    </div>
                    <div>
                        Package Range: ₹{pred['avg_package']:.1f}L - ₹{pred['max_package']:.1f}L
                    </div>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Requirements Status:</strong><br>
                    {requirements_html}
                </div>
                {score_breakdown}
            </div>""", unsafe_allow_html=True)
            
            # Show missing skills details for top predictions
            if i < 5 and (missing_mandatory or missing_preferred):
                with st.expander(f"🔍 Missing Skills for {pred['company_name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if missing_mandatory:
                            st.markdown("**🚨 Missing Mandatory Skills:**")
                            for skill_id in missing_mandatory[:5]:
                                skill_name = predictor.get_skill_name(skill_id)
                                st.markdown(f"• {skill_name}")
                    
                    with col2:
                        if missing_preferred:
                            st.markdown("**⭐ Missing Preferred Skills:**")
                            for skill_id in missing_preferred[:5]:
                                skill_name = predictor.get_skill_name(skill_id)
                                st.markdown(f"• {skill_name}")

def analytics_page(predictor):
    st.header("📊 Analytics & Market Insights")
    
    # Quick insights summary
    st.subheader("📈 Key Market Insights")
    
    # Calculate key insights
    top_tier_companies = len(predictor.data['companies'][predictor.data['companies']['tier'] == 'Tier1'])
    avg_cgpa_requirement = predictor.data['companies']['min_cgpa'].mean()
    most_demanded_skill = predictor.data['skills'].loc[predictor.data['skills']['market_demand_score'].idxmax(), 'skill_name']
    high_package_companies = len(predictor.data['companies'][predictor.data['companies']['avg_package_lpa'] >= 10])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tier 1 Companies", top_tier_companies, help="Premium companies with highest packages")
    with col2:
        st.metric("Avg CGPA Requirement", f"{avg_cgpa_requirement:.1f}", help="Average minimum CGPA across all companies")
    with col3:
        st.metric("Most In-Demand Skill", most_demanded_skill, help="Skill with highest market demand")
    with col4:
        st.metric("High Package Companies", high_package_companies, help="Companies offering ≥10 LPA")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🏢 Company Analysis", "🔧 Skills Intelligence", "📊 Success Patterns"])
    
    with tab1:
        st.subheader("Company Landscape Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Company tier distribution
            tier_counts = predictor.data['companies']['tier'].value_counts()
            fig = px.pie(values=tier_counts.values, names=tier_counts.index, 
                        title="Company Distribution by Tier",
                        color_discrete_map={'Tier1': '#ef4444', 'Tier2': '#f59e0b', 'Tier3': '#22c55e'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Package vs CGPA requirements
            fig = px.scatter(predictor.data['companies'], x='min_cgpa', y='avg_package_lpa',
                            color='tier', size='max_package_lpa',
                            title="Package vs CGPA Requirements",
                            labels={'min_cgpa': 'Minimum CGPA', 'avg_package_lpa': 'Average Package (LPA)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Top companies by package
        st.subheader("💰 Highest Paying Companies")
        top_companies = predictor.data['companies'].nlargest(10, 'avg_package_lpa')[['company_name', 'tier', 'avg_package_lpa', 'min_cgpa', 'location']]
        st.dataframe(top_companies, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Skills Market Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top skills by market demand
            top_skills = predictor.data['skills'].nlargest(10, 'market_demand_score')
            fig = px.bar(top_skills, x='market_demand_score', y='skill_name',
                        orientation='h', title="Top 10 Skills by Market Demand",
                        color='category')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Skills by category
            category_counts = predictor.data['skills']['category'].value_counts()
            fig = px.bar(x=category_counts.index, y=category_counts.values,
                        title="Skills Distribution by Category",
                        color=category_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning time vs market demand
        fig = px.scatter(predictor.data['skills'], x='learning_time_months', y='market_demand_score',
                        color='difficulty_level', size='importance_weight',
                        title="Learning Investment vs Market Demand",
                        labels={'learning_time_months': 'Learning Time (Months)',
                            'market_demand_score': 'Market Demand Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Skills difficulty breakdown
        st.subheader("🎯 Skills Difficulty Analysis")
        difficulty_stats = predictor.data['skills'].groupby('difficulty_level').agg({
            'learning_time_months': 'mean',
            'market_demand_score': 'mean',
            'skill_name': 'count'
        }).round(1)
        difficulty_stats.columns = ['Avg Learning Time (Months)', 'Avg Market Demand', 'Number of Skills']
        st.dataframe(difficulty_stats, use_container_width=True)
    
    with tab3:
        st.subheader("Placement Success Patterns")
        
        # CGPA analysis
        placement_success = predictor.data['placements'].copy()
        placement_success['success'] = placement_success['selection_round_cleared'].map({'Yes': 1, 'No': 0})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CGPA distribution by success
            fig = px.histogram(placement_success, x='student_cgpa', color='selection_round_cleared',
                            title="CGPA Distribution by Placement Success",
                            labels={'student_cgpa': 'Student CGPA', 'count': 'Number of Students'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Branch-wise success rates
            branch_success = placement_success.groupby('branch')['success'].mean().reset_index()
            branch_success['success'] = branch_success['success'] * 100  # Convert to percentage
            
            fig = px.bar(branch_success, x='branch', y='success',
                        title="Placement Success Rate by Branch (%)",
                        color='success', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        # Success factors analysis
        st.subheader("🎯 Key Success Factors")
        success_factors = []
        avg_cgpa_success = placement_success[placement_success['success'] == 1]['student_cgpa'].mean()
        avg_cgpa_fail = placement_success[placement_success['success'] == 0]['student_cgpa'].mean()
        
        avg_internships_success = placement_success[placement_success['success'] == 1]['internship_count'].mean()
        avg_projects_success = placement_success[placement_success['success'] == 1]['project_count'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg CGPA (Successful)", f"{avg_cgpa_success:.2f}", 
                    f"+{avg_cgpa_success - avg_cgpa_fail:.2f}")
        with col2:
            st.metric("Avg Internships (Successful)", f"{avg_internships_success:.1f}")
        with col3:
            st.metric("Avg Projects (Successful)", f"{avg_projects_success:.1f}")
        
        # Recommendations based on analysis
        st.subheader("💡 Data-Driven Recommendations")
        st.markdown(f"""
        **Based on historical placement data:**
        
        - 🎓 **Maintain CGPA above {avg_cgpa_success:.1f}** for better placement chances
        - 💼 **Complete at least {int(avg_internships_success)} internships** before final placements  
        - 🚀 **Work on {int(avg_projects_success)}+ projects** to stand out from competition
        - 🔧 **Focus on {most_demanded_skill}** - it's currently the most in-demand skill
        - 🏢 **Target Tier 2 companies** if your CGPA is below {avg_cgpa_requirement:.1f}
        """)

def recommendations_page(predictor):
    st.header("🎯 Skill Development Recommendations")
    
    # Target companies selection
    st.subheader("🏢 Select Target Companies")
    
    companies_df = predictor.data['companies']
    company_options = companies_df[['company_id', 'company_name', 'tier']].to_dict('records')
    
    selected_companies = st.multiselect(
        "Choose companies you want to target:",
        [f"{comp['company_name']} ({comp['tier']})" for comp in company_options],
        default=[f"{comp['company_name']} ({comp['tier']})" for comp in company_options[:2]]
    )
    
    # Convert back to company IDs
    target_company_ids = []
    for selected in selected_companies:
        for comp in company_options:
            if f"{comp['company_name']} ({comp['tier']})" == selected:
                target_company_ids.append(comp['company_id'])
                break
    
    # Current skills input
    st.subheader("🔧 Your Current Skills")
    
    skills_options = predictor.data['skills'][['skill_id', 'skill_name', 'category']].to_dict('records')
    skill_names = [f"{skill['skill_name']} ({skill['category']})" for skill in skills_options]
    
    current_skills = st.multiselect(
        "Select skills you currently have:",
        skill_names,
        default=skill_names[:2]
    )
    
    # Convert to skill IDs
    current_skill_ids = []
    for selected in current_skills:
        for skill in skills_options:
            if f"{skill['skill_name']} ({skill['category']})" == selected:
                current_skill_ids.append(skill['skill_id'])
                break
    
    if st.button("📋 Get Skill Recommendations", type="primary"):
        student_profile = {
            'skills': current_skill_ids,
            'cgpa': 7.5,  # Default for recommendations
            'branch': 'CSE'
        }
        
        recommendations = predictor.get_skill_recommendations(student_profile, target_company_ids)
        
        if recommendations:
            st.subheader("🚀 Personalized Learning Roadmap")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            high_priority = len([r for r in recommendations if r['priority'] == 'High'])
            total_time = sum(r['learning_time_months'] for r in recommendations[:5])
            avg_difficulty = np.mean([{'High': 3, 'Medium': 2, 'Easy': 1}[r['difficulty']] for r in recommendations])
            avg_demand = np.mean([r['market_demand'] for r in recommendations])
            
            with col1:
                st.metric("High Priority Skills", high_priority)
            with col2:
                st.metric("Est. Learning Time (Top 5)", f"{total_time} months")
            with col3:
                st.metric("Avg Difficulty", f"{avg_difficulty:.1f}/3")
            with col4:
                st.metric("Avg Market Demand", f"{avg_demand:.0f}/100")
            
            # Learning timeline
            st.subheader("📅 Learning Timeline")
            
            timeline_data = []
            cumulative_time = 0
            for i, rec in enumerate(recommendations[:8]):
                timeline_data.append({
                    'skill': rec['skill_name'],
                    'start': cumulative_time,
                    'duration': rec['learning_time_months'],
                    'priority': rec['priority'],
                    'category': rec['category']
                })
                cumulative_time += rec['learning_time_months']
            
            # Create enhanced Gantt chart
            fig = go.Figure()
            
            colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}
            
            for item in timeline_data:
                color = colors.get(item['priority'], '#64748b')
                fig.add_trace(go.Bar(
                    x=[item['duration']],
                    y=[f"{item['skill']} ({item['category']})"],
                    orientation='h',
                    name=item['skill'],
                    base=[item['start']],
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=f"<b>{item['skill']}</b><br>Category: {item['category']}<br>Duration: {item['duration']} months<br>Priority: {item['priority']}<extra></extra>"
                ))
            
            fig.update_layout(
                title="🗓️ Your 8-Skill Learning Journey",
                xaxis_title="Time (Months)",
                yaxis_title="Skills to Learn",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Priority-based skill breakdown
            st.subheader("📚 Detailed Skill Analysis")
            # Separate high and medium priority skills
            high_priority_skills = [r for r in recommendations if r['priority'] == 'High']
            medium_priority_skills = [r for r in recommendations if r['priority'] == 'Medium']
            low_priority_skills = [r for r in recommendations if r['priority'] == 'Low']
            
            if high_priority_skills:
                st.markdown("### 🔥 High Priority Skills (Learn First)")
                for i, rec in enumerate(high_priority_skills, 1):
                    with st.expander(f"🚨 {i}. {rec['skill_name']} - {rec['category']}"):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(f"""
                            **⭐ Priority:** {rec['priority']}  
                            **📊 Difficulty:** {'⭐' * {'High': 3, 'Medium': 2, 'Easy': 1}[rec['difficulty']]} {rec['difficulty']}  
                            **⏱️ Learning Time:** {rec['learning_time_months']} months  
                            **📈 Market Demand:** {rec['market_demand']}/100  
                            **🎯 Importance Score:** {rec['importance_score']}
                            """)
                        with col2:
                            if rec['learning_path']:
                                st.markdown(f"**🛣️ Learning Path:** {rec['learning_path']}")
                            if rec['resources']:
                                st.markdown(f"**📖 Resources:** {rec['resources']}")
                            if 'difficulty_boost' in rec:
                                st.markdown(f"**🚀 Placement Boost:** +{rec['difficulty_boost']}%")
            
            if medium_priority_skills:
                st.markdown("### 🟡 Medium Priority Skills (Learn Next)")
                for i, rec in enumerate(medium_priority_skills, 1):
                    with st.expander(f"📋 {i}. {rec['skill_name']} - {rec['category']}"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **⭐ Priority:** {rec['priority']}  
                            **📊 Difficulty:** {'⭐' * {'High': 3, 'Medium': 2, 'Easy': 1}[rec['difficulty']]} {rec['difficulty']}  
                            **⏱️ Learning Time:** {rec['learning_time_months']} months  
                            **📈 Market Demand:** {rec['market_demand']}/100
                            """)
                        
                        with col2:
                            if rec['learning_path']:
                                st.markdown(f"**🛣️ Learning Path:** {rec['learning_path']}")
                            if rec['resources']:
                                st.markdown(f"**📖 Resources:** {rec['resources']}")
            
            if low_priority_skills:
                st.markdown("### 🟢 Low Priority Skills (Optional)")
                with st.expander(f"📚 {len(low_priority_skills)} Additional Skills"):
                    for rec in low_priority_skills:
                        st.markdown(f"• **{rec['skill_name']}** ({rec['category']}) - {rec['learning_time_months']} months")
            
            # Action plan
            st.subheader("🎯 Your Action Plan")
            next_steps = []
            if high_priority_skills:
                next_steps.append(f"🚨 **Start with:** {high_priority_skills[0]['skill_name']} ({high_priority_skills[0]['learning_time_months']} months)")
            if len(high_priority_skills) > 1:
                next_steps.append(f"🔥 **Then focus on:** {high_priority_skills[1]['skill_name']} ({high_priority_skills[1]['learning_time_months']} months)")
            if medium_priority_skills:
                next_steps.append(f"📝 **Later add:** {medium_priority_skills[0]['skill_name']} for broader opportunities")
            
            for step in next_steps:
                st.markdown(step)
                
        else:
            st.success("🎉 Excellent! You already have most of the required skills for your target companies!")
            st.balloons()
            
            st.markdown("""
            ### 💡 Since you're well-prepared, consider:
            - 🏆 **Advanced certifications** in your existing skills
            - 🚀 **Leadership projects** to stand out
            - 🤝 **Networking** with professionals in target companies
            - 📈 **Staying updated** with latest industry trends""")

if __name__ == "__main__":
    main()