# train_models_actual.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle
import os
import re

class ActualDataModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.label_encoders = {}
    
    def load_datasets(self):
        """Load all datasets"""
        try:
            self.data = {
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
            print("✅ All datasets loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading some datasets: {e}")
            # Load whatever is available
            self.load_available_datasets()
            return True
    
    def load_available_datasets(self):
        """Load only available datasets"""
        files_to_try = [
            ('placements', 'historical placement.csv'),
            ('companies', 'companies dataset.csv'),
            ('skills', 'skills master.csv'),
            ('company_skills', 'company skill.csv')
        ]
        
        for name, filename in files_to_try:
            if os.path.exists(filename):
                try:
                    self.data[name] = pd.read_csv(filename)
                    print(f"✅ Loaded {filename}")
                except Exception as e:
                    print(f"❌ Failed to load {filename}: {e}")
    
    def prepare_training_data_from_placements(self):
        """Prepare training data directly from placements data"""
        print("📊 Preparing training data from placements...")
        
        if 'placements' not in self.data:
            print("❌ Placements data not found!")
            return np.array([]), np.array([])
        
        placements_df = self.data['placements']
        print(f"   Placements data shape: {placements_df.shape}")
        print(f"   Columns: {list(placements_df.columns)}")
        
        features = []
        labels = []
        
        # Process each placement record
        for _, record in placements_df.iterrows():
            try:
                # Extract features directly from placements data
                feature_vector = self.extract_features_from_placement(record)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Since all students are placed, we'll create a regression target instead
                    # Use package_lpa as the target for regression, or create a placement score
                    package = float(record['package_lpa']) if 'package_lpa' in record and pd.notna(record['package_lpa']) else 10.0
                    
                    # Create a placement quality score (0-1) based on package
                    # Normalize package to 0-1 range (assuming packages range from 0-30 LPA)
                    placement_score = min(package / 30.0, 1.0)
                    labels.append(placement_score)
                    
            except Exception as e:
                print(f"   Warning: Error processing record - {e}")
                continue
        
        print(f"   ✅ Prepared {len(features)} training samples")
        print(f"   ✅ Placement scores range: {min(labels):.2f} - {max(labels):.2f}")
        
        return np.array(features), np.array(labels)
    
    def extract_features_from_placement(self, record):
        """Extract features from a placement record"""
        try:
            # CGPA
            cgpa = float(record['student_cgpa'])
            
            # Branch (encode as numeric)
            branch = str(record['branch'])
            if 'branch' not in self.label_encoders:
                self.label_encoders['branch'] = LabelEncoder()
                # Get all unique branches from data
                all_branches = self.data['placements']['branch'].unique() if 'placements' in self.data else [branch]
                self.label_encoders['branch'].fit(all_branches)
            branch_encoded = self.label_encoders['branch'].transform([branch])[0]
            
            # Skills count
            skills_str = str(record['skills_possessed'])
            skills_count = self.count_skills(skills_str)
            
            # Internship count
            internship_count = int(record['internship_count'])
            
            # Project count
            project_count = int(record['project_count'])
            
            # Certifications (binary - has or doesn't have)
            certs = str(record['certifications'])
            has_certifications = 1 if certs.lower() not in ['', 'na', 'none', 'no', '0'] else 0
            
            # Company features (if company data is available)
            company_name = record['company_placed'] if 'company_placed' in record else 'Unknown'
            company_tier = self.get_company_tier(company_name)
            min_cgpa = self.get_company_min_cgpa(company_name)
            
            # Create feature vector
            feature_vector = [
                cgpa,
                branch_encoded,
                skills_count,
                internship_count,
                project_count,
                has_certifications,
                company_tier,
                min_cgpa
            ]
            
            return feature_vector
            
        except Exception as e:
            print(f"   Error extracting features: {e}")
            return None
    
    def count_skills(self, skills_str):
        """Count number of skills from string"""
        if pd.isna(skills_str) or skills_str == '':
            return 0
        
        try:
            # Handle different formats
            if ',' in skills_str:
                return len([s.strip() for s in skills_str.split(',') if s.strip()])
            elif ';' in skills_str:
                return len([s.strip() for s in skills_str.split(';') if s.strip()])
            else:
                return 1 if skills_str.strip() else 0
        except:
            return 0
    
    def get_company_tier(self, company_name):
        """Get company tier from company data"""
        if 'companies' not in self.data or pd.isna(company_name):
            return 2  # Default to tier 2
        
        try:
            company_col = None
            for col in self.data['companies'].columns:
                if 'company' in col.lower() or 'name' in col.lower():
                    company_col = col
                    break
            
            if company_col:
                company_data = self.data['companies'][self.data['companies'][company_col].str.contains(company_name, case=False, na=False)]
                if len(company_data) > 0:
                    tier = company_data.iloc[0].get('tier', 'Tier2')
                    if 'tier1' in str(tier).lower():
                        return 1
                    elif 'tier3' in str(tier).lower():
                        return 3
            return 2
        except:
            return 2
    
    def get_company_min_cgpa(self, company_name):
        """Get company minimum CGPA from company data"""
        if 'companies' not in self.data or pd.isna(company_name):
            return 7.0  # Default
        
        try:
            company_col = None
            for col in self.data['companies'].columns:
                if 'company' in col.lower() or 'name' in col.lower():
                    company_col = col
                    break
            
            if company_col:
                company_data = self.data['companies'][self.data['companies'][company_col].str.contains(company_name, case=False, na=False)]
                if len(company_data) > 0:
                    return float(company_data.iloc[0].get('min_cgpa', 7.0))
            return 7.0
        except:
            return 7.0
    
    def train_regression_models(self, X, y):
        """Train regression models since we have continuous placement scores"""
        print("🤖 Training regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['placement'] = scaler
        
        # Since we're doing regression, let's use different models
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        print("   Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        print("   Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model
        
        print("   Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        self.models['linear_regression'] = lr_model
        
        # Evaluate models
        print("\n📈 Model Evaluation (R² Score & Mean Squared Error):")
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"   {name}: R² = {r2:.3f}, MSE = {mse:.3f}")
        
        return True
    
    def train_models(self):
        """Train machine learning models"""
        print("🤖 Training models on actual placements data...")
        
        # Prepare training data
        X, y = self.prepare_training_data_from_placements()
        
        if len(X) == 0 or len(y) == 0:
            print("❌ No training data available!")
            return False
        
        print(f"   Training set shape: {X.shape}")
        
        # Since all samples are placed, we'll train regression models
        # to predict placement quality (package-based score)
        return self.train_regression_models(X, y)
    
    def save_models(self):
        """Save trained models to files"""
        print("💾 Saving models...")
        
        try:
            # Save models
            with open('random_forest_model.pkl', 'wb') as f:
                pickle.dump(self.models['random_forest'], f)
            
            with open('gradient_boosting_model.pkl', 'wb') as f:
                pickle.dump(self.models['gradient_boosting'], f)
            
            with open('linear_regression_model.pkl', 'wb') as f:
                pickle.dump(self.models['linear_regression'], f)
            
            # Save scaler
            with open('placement_scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers['placement'], f)
            
            # Save label encoders if needed
            with open('label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            print("✅ Models saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False

def main():
    """Main training function"""
    print("🚀 Starting Model Training with Actual Data")
    print("=" * 50)
    
    trainer = ActualDataModelTrainer()
    
    if not trainer.load_datasets():
        print("❌ Failed to load datasets!")
        return
    
    if trainer.train_models():
        if trainer.save_models():
            print("🎉 Training completed successfully with REAL data!")
            print("✅ Your models are now properly trained to predict placement QUALITY!")
        else:
            print("⚠️  Training completed but models couldn't be saved!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()