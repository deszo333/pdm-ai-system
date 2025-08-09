# ENHANCED SMART STUDENT DATA SYSTEM (FINAL VERSION)
# Universal data extraction with smart hierarchical organization, debug toggle, and API switcher.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import warnings
import pandas as pd
import os
import fitz
import re
from datetime import datetime
import json
import requests
import ollama # Re-added for offline mode

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration for Online Mistral API ---
# This is only used when the API mode is 'online'
MISTRAL_API_KEY = "fcbJyUY4pHwpCNOTB7Wq3IZaivGdzz01" # Replace with your key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


### NEW: UNIVERSAL LLM CALLER ###
def call_llm(prompt_text, sds):
    """
    Calls the appropriate LLM (online or offline) based on the current system setting.
    """
    if sds.api_mode == 'online':
        headers = {
            'Authorization': f'Bearer {MISTRAL_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        data = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt_text}]}
        try:
            response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed. {e}"
        except (KeyError, IndexError):
            return "Error: Could not parse the API response."

    elif sds.api_mode == 'offline':
        try:
            response = ollama.chat(model="mistral:instruct", messages=[
                {"role": "user", "content": prompt_text}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error: Offline model call failed. Is Ollama running? Details: {e}"

def extract_filters_from_query(query, sds):
    """
    Asks the AI to perform a simple extraction of key-value pairs.
    Now uses the universal call_llm function.
    """
    possible_fields = ["year_level", "course", "section", "name"]
    prompt = f"""
    You are an entity extraction assistant. Your task is to extract information from the user's query into a simple JSON object.

    **Possible Keys:** {json.dumps(possible_fields)}

    **Rules:**
    - If you see a course, year, or section, extract it.
    - If you see something that looks like a person's name (one or more words), extract the entire name string into the "name" key.
    - Convert all values to lowercase, except for course codes (e.g., "BSCS").

    Query: "{query}"
    """
    response_content = call_llm(prompt, sds)
    try:
        start_index = response_content.find('{')
        end_index = response_content.rfind('}')
        if start_index != -1 and end_index != -1:
            json_string = response_content[start_index : end_index + 1]
            filters = json.loads(json_string)
        else:
            raise ValueError("No valid JSON object found in the AI response.")

        if sds.debug_mode:
            print("\n🔍 Parsed Filters from Mistral (Simple Format):")
            print(json.dumps(filters, indent=2))
        return filters
    except Exception as e:
        if sds.debug_mode:
            print("❌ Failed to parse filters:", e)
            print(f"   Raw Response from API: {response_content}")
        return {}

def summarize_results_with_mistral(user_query, results, sds):
    """
    Generates a final answer with the hardened prompt to prevent hallucination.
    Now uses the universal call_llm function.
    """
    if not results:
        return "I could not find any records matching your query."
    
    unique_contents = {r.get('id'): r for r in results}.values()
    combined_docs = "\n\n".join(f"Source File: {r['metadata'].get('source_file', 'N/A')}\nRecord:\n{r.get('content', '')}" for r in unique_contents)

    prompt = f"""
    You are a database retrieval assistant. Your ONLY function is to report on the data provided in the "Relevant Student Data" section. You MUST NOT use any external or pre-existing knowledge.

    **CRITICAL RULE:** If the "Relevant Student Data" section below is empty, you MUST respond with ONLY the following message: "I could not find any records matching your query." Do not add any other text.

    If data IS provided, answer the user's question based ONLY on that data.

    User's Question: "{user_query}"
    Relevant Student Data:
    {combined_docs}
    """
    return call_llm(prompt, sds)

def rag_query_pipeline(sds, user_query):
    """
    The final RAG pipeline, now with debug prints conditional.
    """
    if not sds.check_existing_data(silent=True):
        return "❌ No data loaded. Please upload or process data first."

    simple_filters = extract_filters_from_query(user_query, sds)
    
    name_query_term = simple_filters.pop('name', None)
    where_clause = simple_filters
    
    if 'course' in where_clause and isinstance(where_clause.get('course'), str):
        where_clause['course'] = where_clause['course'].upper()
    if len(where_clause) > 1:
        where_clause = {"$and": [{k: v} for k, v in where_clause.items()]}

    results = sds.smart_search_with_filters(user_query, name_query_term, where_clause, max_results=100)

    if sds.debug_mode:
        print("\n📚 Retrieved Relevant Results:")
        for r in results:
            source_file = r.get('metadata', {}).get('source_file', 'N/A')
            print(f"- [{source_file}] {r.get('content', '').strip()}")

    return summarize_results_with_mistral(user_query, results, sds)

class SmartStudentDataSystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_store")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collections = {}
        self.data_loaded = False
        
        # --- NEW SETTINGS ---
        self.debug_mode = True  # Set to False to hide detailed logs
        self.api_mode = 'online' # Can be 'online' or 'offline'
        
    # ======================== INITIALIZATION & SETUP ========================
    
    def check_existing_data(self, silent=False):
        """Check if there's already data in ChromaDB."""
        try:
            existing_collections = self.client.list_collections()
            if existing_collections:
                if not silent:
                    print("🗃️ Found existing data in ChromaDB:")
                    for i, collection in enumerate(existing_collections, 1):
                        count = collection.count()
                        collection_type = self.get_collection_type(collection.name)
                        print(f"  {i}. {collection.name} - {collection_type} ({count} records)")
                return existing_collections
            return []
        except:
            return []
    
    def get_collection_type(self, name):
        """Get human-readable collection type from collection name"""
        # Smart collection names like: students_ccs_bscs_year2_seca
        parts = name.split('_')
        
        if len(parts) >= 2:
            base_type = parts[0]
            dept = parts[1] if len(parts) > 1 else ""
            course = parts[2] if len(parts) > 2 else ""
            year = parts[3] if len(parts) > 3 else ""
            section = parts[4] if len(parts) > 4 else ""
            
            if base_type == "students":
                return f"Students - {dept.upper()} {course.upper()} {year} {section.upper()}".strip()
            elif base_type == "schedules":
                return f"COR Schedule - {dept.upper()} {course.upper()} {year} {section.upper()}".strip()
            elif base_type == "faculty":
                faculty_type = parts[-1] if len(parts) > 2 else "general"
                return f"Faculty {faculty_type.title()} - {dept.upper()}".strip()
        
        # Fallback to old naming
        type_map = {
            "students_excel": "Student Data (Excel)",
            "students_pdf": "Student Data (PDF)", 
            "schedules_excel": "COR Schedules (Excel)",
            "schedules_pdf": "COR Schedules (PDF)",
            "faculty_excel": "Faculty Data (Excel)",
            "faculty_pdf": "Faculty Data (PDF)"
        }
        return type_map.get(name, f"Data Collection ({name})")
    
    def quick_setup(self):
        """Quick setup - check existing data or load new"""
        existing = self.check_existing_data()
        
        if existing:
            print(f"\n🚀 Ready to query! Found {len(existing)} data collections.")
            print("💡 You can search across all your data immediately.")
            self.data_loaded = True
            
            # Load all existing collections
            for collection in existing:
                self.collections[collection.name] = collection
            
            return True
        else:
            print("📂 No existing data found. Let's load some files first...")
            return self.load_new_data()
    
    # ======================== FILE MANAGEMENT ========================
    
    def list_available_files(self):
        """List available files with smart type detection"""
        files = [f for f in os.listdir('.') 
                if (f.endswith('.xlsx') or f.endswith('.pdf')) and not f.startswith('~$')]
        
        if not files:
            print("❌ No Excel or PDF files found.")
            return []
        
        print("\n📁 Available Files:")
        for i, file in enumerate(files, 1):
            file_type = self.detect_file_type(file)
            print(f"  {i}. {file} - {file_type}")
        return files
    
    def detect_file_type(self, filename):
        """Smart file type detection"""
        ext = os.path.splitext(filename)[1].lower()
        filename_lower = filename.lower()
        
        if ext == ".xlsx":
            try:
                # Check filename patterns first
                if any(x in filename_lower for x in ['resume', 'cv', 'faculty_data']):
                    return "Faculty Data (Excel)"
                elif 'cor' in filename_lower:
                    return "COR Schedule (Excel)"
                elif any(x in filename_lower for x in ['schedule', 'class_schedule']):
                    return "Faculty Schedule (Excel)"
                elif any(x in filename_lower for x in ['student', 'year', 'bscs', 'bsit', 'bstm', 'bshm']):
                    return "Student Data (Excel)"
                
                # Check content if filename is unclear
                df_check = pd.read_excel(filename, header=None)
                if self.is_cor_file(df_check):
                    return "COR Schedule (Excel)"
                elif self.is_faculty_schedule_excel(df_check):
                    return "Faculty Schedule (Excel)"
                elif self.is_faculty_excel(df_check):
                    return "Faculty Data (Excel)"
                else:
                    return "Student Data (Excel)"
            except:
                return "Excel File"
                
        elif ext == ".pdf":
            # Check filename patterns
            if any(x in filename_lower for x in ['resume', 'cv']):
                return "Faculty Data (PDF)"
            elif 'cor' in filename_lower:
                return "COR Schedule (PDF)"
            elif 'schedule' in filename_lower:
                return "Faculty Schedule (PDF)"
            elif any(x in filename_lower for x in ['student', 'year', 'synthetic']):
                return "Student Data (PDF)"
                
            # Check content
            if self.is_cor_pdf(filename):
                return "COR Schedule (PDF)"
            elif self.is_faculty_schedule_pdf(filename):
                return "Faculty Schedule (PDF)"
            elif self.is_faculty_pdf(filename):
                return "Faculty Data (PDF)"
            else:
                return "Student Data (PDF)"
        
        return "Unknown"
    
    # ======================== UNIVERSAL DATA EXTRACTION ========================
    
    def extract_universal_student_data(self, text_content, source_type):
        """Universal extractor for student data regardless of format"""
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        all_text = ' '.join(lines).upper()
        
        # Initialize required fields
        student_data = {
            'student_id': None,
            'surname': None,
            'first_name': None,
            'full_name': None,
            'year': None,
            'course': None,
            'section': None,
            'contact_number': None,
            'guardian_name': None,
            'guardian_contact': None
        }
        
        # For structured Excel, extract directly from column headers
        if source_type == 'excel_structured':
            return self.extract_from_structured_text(lines)
        
        # Rest of the existing patterns for unstructured data...
        patterns = {
            'student_id': [
                r'STUDENT\s*ID[:\s]*([A-Z0-9-]+)',
                r'ID\s*NO[:\s]*([A-Z0-9-]+)',
                r'ID[:\s]*([A-Z0-9-]+)',
                r'STUDENT\s*NUMBER[:\s]*([A-Z0-9-]+)',
                r'([A-Z]{2,4}-\d{4,6})',  # Pattern like PDM-123456
                r'(\d{4,8})',  # Pure numbers
            ],
            # ... rest of patterns remain the same
        }
        
        # Extract each field using patterns
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    value = matches[0].strip()
                    if value and value not in ['', 'N/A', 'NULL', 'NONE', 'SURNAME', 'FIRST NAME', 'GUARDIAN NAME']:
                        student_data[field] = self.clean_extracted_value(value, field)
                        break
            
            # If not found, try fuzzy matching in lines
            if not student_data[field]:
                student_data[field] = self.fuzzy_field_extraction(lines, field)
        
        # Post-process name splitting
        if student_data['name'] and not (student_data['surname'] and student_data['first_name']):
            student_data['surname'], student_data['first_name'] = self.split_full_name(student_data['name'])
        
        student_data['full_name'] = student_data['name']
        
        return student_data

    def clean_extracted_value(self, value, field_type):
        """Clean and validate extracted values"""
        if not value:
            return None
        
        value = value.strip()
        
        # Filter out common header values
        header_values = ['SURNAME', 'FIRST NAME', 'GUARDIAN NAME', 'CONTACT NUMBER', 'STUDENT ID', 'YEAR', 'COURSE', 'SECTION']
        if value.upper() in header_values:
            return None
        
        if field_type == 'student_id':
            # Keep only alphanumeric and dashes
            cleaned = re.sub(r'[^A-Z0-9-]', '', value)
            return cleaned if cleaned else None
        
        elif field_type in ['contact_number', 'guardian_contact']:
            # Clean phone numbers
            cleaned = re.sub(r'[^\d\+]', '', value)
            if len(cleaned) >= 10:
                return cleaned
        
        elif field_type in ['name', 'guardian_name', 'surname', 'first_name']:
            # Clean names - keep letters, spaces, dots, commas
            cleaned = re.sub(r'[^A-Za-z\s\.,]', '', value).title()
            return cleaned if cleaned and len(cleaned) > 1 else None
        
        elif field_type == 'year':
            # Extract just the number and convert it to an integer
            year_match = re.search(r'([1-4])', value)
            if year_match:
                return int(year_match.group(1)) # <-- THE FIX IS HERE
            return None
        
        elif field_type in ['course', 'section']:
            # Keep uppercase letters and numbers
            cleaned = re.sub(r'[^A-Z0-9]', '', value)
            return cleaned if cleaned else None
        
        return value

    def fuzzy_field_extraction(self, lines, field_type):
        """Fuzzy extraction when pattern matching fails"""
        field_keywords = {
            'student_id': ['id', 'student', 'number'],
            'name': ['name'],
            'year': ['year', 'level'],
            'course': ['course', 'program'],
            'section': ['section', 'class'],
            'contact_number': ['contact', 'phone', 'mobile'],
            'guardian_name': ['guardian', 'parent', 'emergency'],
            'guardian_contact': ['guardian', 'parent', 'emergency']
        }
        
        keywords = field_keywords.get(field_type, [])
        
        for line in lines:
            line_upper = line.upper()
            if any(keyword.upper() in line_upper for keyword in keywords):
                # Try to extract value after colon or space
                if ':' in line:
                    value = line.split(':', 1)[1].strip()
                else:
                    # Extract the part that looks like the data we want
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if any(keyword.upper() in part.upper() for keyword in keywords):
                            if i + 1 < len(parts):
                                value = parts[i + 1]
                                break
                    else:
                        continue
                
                if value and len(value) > 1:
                    return self.clean_extracted_value(value, field_type)
        
        return None

    def split_full_name(self, full_name):
        """Split full name into surname and first name"""
        if not full_name:
            return None, None
        
        name_parts = full_name.strip().split()
        
        if len(name_parts) == 1:
            return name_parts[0], None
        elif len(name_parts) == 2:
            return name_parts[0], name_parts[1]  # Assuming "Surname FirstName"
        else:
            # If comma exists, assume "Surname, FirstName" format
            if ',' in full_name:
                parts = full_name.split(',', 1)
                surname = parts[0].strip()
                first_name = parts[1].strip()
                return surname, first_name
            else:
                # Assume last word is surname, rest is first name
                return name_parts[-1], ' '.join(name_parts[:-1])

    def split_into_student_records(self, text):
        """Split text into individual student records"""
        # Try different splitting strategies
        
        # Strategy 1: Split by student ID patterns
        id_patterns = [r'[A-Z]{2,4}-\d{4,6}', r'PDM-\d+', r'STUDENT\s*ID']
        
        for pattern in id_patterns:
            splits = re.split(f'({pattern})', text, flags=re.IGNORECASE)
            if len(splits) > 3:  # Found multiple student records
                records = []
                for i in range(1, len(splits), 2):  # Every other element starting from 1
                    if i + 1 < len(splits):
                        record = splits[i] + splits[i + 1]
                        records.append(record)
                return records
        
        # Strategy 2: Split by line breaks and group by proximity
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        records = []
        current_record = []
        
        for line in lines:
            # If line contains student ID pattern, start new record
            if re.search(r'[A-Z]{2,4}-\d{4,6}|PDM-\d+|STUDENT\s*ID', line, re.IGNORECASE):
                if current_record:
                    records.append('\n'.join(current_record))
                current_record = [line]
            else:
                current_record.append(line)
                
                # If record gets too long, close it
                if len(current_record) > 10:
                    records.append('\n'.join(current_record))
                    current_record = []
        
        if current_record:
            records.append('\n'.join(current_record))
        
        # If still no good splits, return whole text as one record
        return records if records else [text]

    # ======================== SMART HIERARCHY & METADATA ========================
    
    def extract_smart_metadata(self, text, file_type):
        """Enhanced smart extraction of organizational metadata from text"""
        metadata = {
            'course': None,
            'section': None, 
            'year_level': None,
            'department': None,
            'faculty_type': None,
            'data_type': file_type,
            'subject_codes': []
        }
        
        text_upper = text.upper()
        
        # Enhanced course detection
        course_patterns = [
            r'COURSE[:\s]*([A-Z]{2,6})',
            r'PROGRAM[:\s]*([A-Z]{2,6})',
            r'BS[A-Z]{2,4}',
            r'AB[A-Z]*',
            r'BA[A-Z]*'
        ]
        
        for pattern in course_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['course'] = matches[0] if isinstance(matches[0], str) else matches[0]
                break
        
        # Enhanced section detection
        section_patterns = [
            r'SECTION[:\s]*([A-Z0-9-]+)',
            r'SEC[:\s]*([A-Z0-9-]+)',
            r'SECTION\s+([A-Z0-9-]+)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['section'] = matches[0]
                break
        
        # Enhanced year level detection
        year_patterns = [
            r'YEAR\s*LEVEL[:\s]*([1-4])',
            r'YEAR[:\s]*([1-4])',
            r'([1-4])(?:ST|ND|RD|TH)?\s*YEAR'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                metadata['year_level'] = matches[0]
                break
        
        # Subject code extraction for schedules
        subject_matches = re.findall(r'[A-Z]{2,4}\d{3}', text_upper)
        metadata['subject_codes'] = list(set(subject_matches))  # Remove duplicates
        
        # Smart department detection from course
        if metadata['course']:
            metadata['department'] = self.detect_department_from_course(metadata['course'])
        
        # Faculty type detection (enhanced)
        if 'faculty' in file_type.lower():
            if any(keyword in text.lower() for keyword in ['schedule', 'class schedule', 'teaching']):
                metadata['faculty_type'] = 'teaching'
            elif any(keyword in text.lower() for keyword in ['resume', 'cv', 'profile']):
                metadata['faculty_type'] = 'profile'
            else:
                metadata['faculty_type'] = 'general'
        
        return metadata
    
    def contextual_department_inference(self, course_upper):
        """Infer department from course structure and common patterns"""
        
        # Analyze degree prefix patterns
        if course_upper.startswith('BS'):
            suffix = course_upper[2:]  # Remove 'BS'
            
            # Technology indicators
            if any(indicator in suffix for indicator in ['TECH', 'DIGITAL', 'COMP', 'INFO', 'DATA', 'CYBER', 'SOFT', 'WEB']):
                return 'CCS'
            
            # Engineering indicators
            elif any(indicator in suffix for indicator in ['ENG', 'MECH', 'ELEC', 'CIV', 'IND', 'CHEM', 'AERO']):
                return 'COE'
            
            # Business indicators
            elif any(indicator in suffix for indicator in ['BUS', 'ADMIN', 'MANAGE', 'ACCT', 'FIN', 'MARKET', 'ECON']):
                return 'CBA'
            
            # Hospitality indicators
            elif any(indicator in suffix for indicator in ['HOST', 'TOUR', 'HOTEL', 'CULINARY', 'REST', 'FOOD']):
                return 'CHTM'
            
            # Health indicators
            elif any(indicator in suffix for indicator in ['NURS', 'HEALTH', 'MED', 'CARE', 'THERAPY']):
                return 'CON'
            
            # Education indicators
            elif any(indicator in suffix for indicator in ['ED', 'TEACH', 'CHILD', 'ELEM', 'SEC', 'SPEC']):
                return 'CED'
        
        elif course_upper.startswith('AB'):
            return 'CAS'  # Liberal arts
        
        elif course_upper.startswith('MA') or course_upper.startswith('MS'):
            # Master's degrees - try to infer from content
            if any(ed in course_upper for ed in ['ED', 'TEACH']):
                return 'CED'
            elif any(bus in course_upper for bus in ['BUS', 'ADMIN', 'MBA']):
                return 'CBA'
            elif any(tech in course_upper for tech in ['COMP', 'INFO', 'TECH']):
                return 'CCS'
        
        return 'UNKNOWN'

    def create_intelligent_category(self, course_upper):
        """Create intelligent category names for unknown courses"""
        
        # Try to create meaningful category based on course characteristics
        if course_upper.startswith('BS'):
            return 'EMERGING_BS'  # Bachelor of Science - Emerging Program
        elif course_upper.startswith('AB'):
            return 'EMERGING_AB'  # Bachelor of Arts - Emerging Program  
        elif course_upper.startswith('MA'):
            return 'GRADUATE_MA'  # Master of Arts Program
        elif course_upper.startswith('MS'):
            return 'GRADUATE_MS'  # Master of Science Program
        elif course_upper.startswith('PHD') or course_upper.startswith('DR'):
            return 'DOCTORAL'     # Doctoral Program
        else:
            return 'NEW_PROGRAM'  # Completely new program type

    def get_department_display_name(self, dept_code):
        """Get human-readable department names including intelligent categories"""
        dept_names = {
            'CCS': 'College of Computer Studies',
            'COE': 'College of Engineering', 
            'CHTM': 'College of Hospitality & Tourism Management',
            'CBA': 'College of Business Administration',
            'CED': 'College of Education',
            'CAS': 'College of Arts & Sciences',
            'CON': 'College of Nursing',
            'EMERGING_BS': 'Emerging BS Program',
            'EMERGING_AB': 'Emerging AB Program', 
            'GRADUATE_MA': 'Graduate MA Program',
            'GRADUATE_MS': 'Graduate MS Program',
            'DOCTORAL': 'Doctoral Program',
            'NEW_PROGRAM': 'New Academic Program',
            'UNKNOWN': 'Unclassified Program'
        }
        return dept_names.get(dept_code, dept_code)

    def detect_department_from_course(self, course):
        """Smart department detection with known courses first, then AI reasoning"""
        if not course:
            return 'UNKNOWN'
        
        course_upper = course.upper()
        
        # Layer 1: Check your known courses FIRST (exact matches)
        known_courses = {
            'CCS': ['BSCS', 'BSIT'],
            'CHTM': ['BSHM', 'BSTM'], 
            'CBA': ['BSOA'],
            'CED': ['BECED', 'BTLE']
        }
        
        # Exact match check for known courses
        for department, courses in known_courses.items():
            if course_upper in courses:
                return department
        
        # Layer 2: Only if NOT in known courses, use AI reasoning
        # Multi-layered smart detection for NEW/UNKNOWN courses
        department_intelligence = {
            'CCS': {
                'primary_domains': ['computer', 'information', 'technology', 'software', 'digital'],
                'technical_terms': ['programming', 'system', 'network', 'data', 'cyber', 'web', 'mobile', 'ai', 'ml', 'robotics'],
                'degree_patterns': ['cs', 'it', 'is', 'se', 'ds', 'ai', 'cyber', 'game', 'multimedia'],
                'keywords': ['computing', 'informatics', 'tech', 'digital', 'automation', 'algorithm']
            },
            'COE': {
                'primary_domains': ['engineering', 'technical', 'mechanical', 'electrical', 'civil'],
                'technical_terms': ['design', 'construction', 'manufacturing', 'power', 'structure', 'material', 'energy'],
                'degree_patterns': ['ce', 'me', 'ee', 'ie', 'che', 'ece', 'ae', 'pe', 'engr'],
                'keywords': ['engineer', 'technical', 'industrial', 'mechanical', 'electrical', 'chemical', 'environmental']
            },
            'CHTM': {
                'primary_domains': ['hospitality', 'tourism', 'service', 'management', 'culinary'],
                'technical_terms': ['hotel', 'restaurant', 'travel', 'event', 'catering', 'resort', 'guest'],
                'degree_patterns': ['hm', 'tm', 'culinary', 'resto', 'event', 'cruise', 'airline'],
                'keywords': ['hospitality', 'tourism', 'hotel', 'culinary', 'service', 'leisure', 'recreation']
            },
            'CBA': {
                'primary_domains': ['business', 'administration', 'management', 'finance', 'commerce'],
                'technical_terms': ['accounting', 'marketing', 'economics', 'entrepreneurship', 'banking', 'trade'],
                'degree_patterns': ['ba', 'bsa', 'bsba', 'oa', 'acct', 'fin', 'mktg', 'econ', 'entrep'],
                'keywords': ['business', 'management', 'admin', 'finance', 'accounting', 'marketing', 'economics']
            },
            'CED': {
                'primary_domains': ['education', 'teaching', 'learning', 'pedagogy', 'instruction'],
                'technical_terms': ['elementary', 'secondary', 'special', 'early', 'childhood', 'curriculum', 'assessment'],
                'degree_patterns': ['ed', 'ece', 'sped', 'tle', 'pe', 'teach', 'elem', 'sec'],
                'keywords': ['education', 'teaching', 'pedagogy', 'learning', 'instruction', 'curriculum', 'child']
            },
            'CAS': {
                'primary_domains': ['arts', 'sciences', 'liberal', 'humanities', 'social'],
                'technical_terms': ['psychology', 'sociology', 'communication', 'literature', 'philosophy', 'history'],
                'degree_patterns': ['ab', 'psych', 'socio', 'comm', 'lit', 'phil', 'hist', 'pol'],
                'keywords': ['arts', 'science', 'liberal', 'social', 'humanities', 'culture', 'society']
            },
            'CON': {
                'primary_domains': ['nursing', 'health', 'medical', 'clinical', 'care'],
                'technical_terms': ['patient', 'healthcare', 'clinical', 'medical', 'therapeutic', 'wellness'],
                'degree_patterns': ['bsn', 'rn', 'health', 'medical', 'clinical', 'therapy'],
                'keywords': ['nursing', 'health', 'medical', 'care', 'clinical', 'patient', 'wellness']
            }
        }
        
        # Smart scoring for unknown courses only
        best_matches = []
        
        for dept, intelligence in department_intelligence.items():
            total_score = 0
            
            # Score matches
            for domain in intelligence['primary_domains']:
                if domain in course_upper:
                    total_score += 20
            
            for term in intelligence['technical_terms']:
                if term in course_upper:
                    total_score += 10
            
            for pattern in intelligence['degree_patterns']:
                if pattern in course_upper:
                    total_score += 25
            
            for keyword in intelligence['keywords']:
                if keyword in course_upper:
                    total_score += 5
            
            if total_score > 0:
                best_matches.append({'department': dept, 'score': total_score})
        
        # Return best match for unknown courses
        if best_matches:
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = best_matches[0]
            
            if best_match['score'] >= 10:
                return best_match['department']
        
        # Contextual inference for completely new courses
        contextual_dept = self.contextual_department_inference(course_upper)
        if contextual_dept != 'UNKNOWN':
            return contextual_dept
        
        # Only create "emerging" category for truly unknown courses
        return self.create_intelligent_category(course_upper)

    def create_smart_collection_name(self, base_type, metadata):
        """Create hierarchical collection name based on metadata"""
        parts = [base_type]
        
        # Smart department detection
        dept = metadata.get('department', '').upper()
        if dept and dept != 'UNKNOWN':
            parts.append(dept.lower())
        
        # Add course
        course = metadata.get('course', '')
        if course:
            parts.append(course.lower().replace(' ', '_'))
        
        # Add year level
        year_level = metadata.get('year_level', '')
        if year_level:
            parts.append(f"year{year_level}")
        
        # Add section
        section = metadata.get('section', '')
        if section:
            parts.append(f"sec{section.lower()}")
        
        # Add faculty type for faculty data
        if base_type == 'faculty':
            faculty_type = metadata.get('faculty_type', 'general')
            parts.append(faculty_type)
        
        return "_".join(parts)

    def store_with_smart_hierarchy(self, texts, metadata_list, base_type):
        """Store data with smart hierarchical organization"""
        try:
            # Group data by smart hierarchy
            hierarchy_groups = {}
            
            for text, metadata in zip(texts, metadata_list):
                # Create smart collection name based on hierarchy
                collection_name = self.create_smart_collection_name(base_type, metadata)
                
                if collection_name not in hierarchy_groups:
                    hierarchy_groups[collection_name] = {
                        'texts': [], 
                        'metadata': [],
                        'sample_meta': metadata
                    }
                
                hierarchy_groups[collection_name]['texts'].append(text)
                hierarchy_groups[collection_name]['metadata'].append(metadata)
            
            # Store each group in its own collection
            success_count = 0
            for collection_name, group_data in hierarchy_groups.items():
                try:
                    collection = self.client.get_or_create_collection(name=collection_name)
                    self.store_with_smart_metadata(collection, group_data['texts'], group_data['metadata'])
                    self.collections[collection_name] = collection
                    
                    # Display organization info
                    sample = group_data['sample_meta']
                    hierarchy_path = f"{sample.get('department', 'Unknown')} > {sample.get('course', 'Unknown')} > Year {sample.get('year_level', 'Unknown')} > Section {sample.get('section', 'Unknown')}"
                    
                    print(f"✅ Stored {len(group_data['texts'])} records in: {collection_name}")
                    print(f"📁 Hierarchy: {hierarchy_path}")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error storing {collection_name}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error in smart hierarchy storage: {e}")
            return False

    def store_with_smart_metadata(self, collection, texts, metadata_list):
        """Store embeddings with rich metadata for smart filtering"""
        for idx, (text, metadata) in enumerate(zip(texts, metadata_list)):
            embedding = self.model.encode(text).tolist()
            
            # Create unique ID with metadata info
            doc_id = f"{metadata.get('course', 'unknown')}_{metadata.get('section', 'unknown')}_{idx}_{datetime.now().timestamp()}"
            
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],  # Store metadata for filtering
                ids=[doc_id]
            )

    # ======================== STUDENT DATA PROCESSING ========================
    
    def process_student_excel(self, filename):
        """Process Student Excel file with universal extraction"""
        try:
            # Try to read as structured data first
            df = pd.read_excel(filename)
            
            if self.is_structured_student_data(df):
                return self.process_structured_student_excel(df, filename)
            else:
                return self.process_unstructured_student_excel(filename)
        
        except Exception as e:
            print(f"❌ Error processing student Excel: {e}")
            return False

    def is_structured_student_data(self, df):
        """Check if Excel has structured column headers"""
        if df.empty:
            return False
        
        # Check if columns contain expected student data headers
        columns_upper = [str(col).upper() for col in df.columns]
        expected_fields = ['STUDENT', 'NAME', 'YEAR', 'COURSE', 'SECTION', 'CONTACT']
        
        matches = sum(1 for field in expected_fields 
                     if any(field in col for col in columns_upper))
        
        return matches >= 4  # At least 4 expected fields found

    def process_structured_student_excel(self, df, filename):
        """Process structured Excel with clear columns"""
        texts = []
        metadata_list = []
        
        # Skip the header row and process actual data
        for idx, row in df.iterrows():
            if idx == 0:  # Skip header row
                continue
                
            # Convert row to text for universal extraction
            row_text = ""
            for col, value in row.items():
                if pd.notna(value) and str(value).strip() and str(value).strip() != 'nan':
                    row_text += f"{col}: {value}\n"
            
            # Use universal extractor
            student_data = self.extract_universal_student_data(row_text, 'excel_structured')
            
            if student_data['student_id']:  # Only process if we found a student ID
                formatted_text = self.format_student_data(student_data)
                metadata = self.create_student_metadata(student_data)
                
                texts.append(formatted_text)
                metadata_list.append(metadata)
        
        if texts:
            return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
        else:
            print("❌ No valid student data found")
            return False

    def process_unstructured_student_excel(self, filename):
        """Process unstructured Excel data"""
        try:
            # Read all sheets and all data as text
            xl_file = pd.ExcelFile(filename)
            all_text = ""
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
                for row in df.values:
                    row_text = ' '.join([str(cell) for cell in row if pd.notna(cell)])
                    all_text += row_text + "\n"
            
            # Split into potential student records
            student_records = self.split_into_student_records(all_text)
            
            texts = []
            metadata_list = []
            
            for record in student_records:
                student_data = self.extract_universal_student_data(record, 'excel_unstructured')
                
                if student_data['student_id']:
                    formatted_text = self.format_student_data(student_data)
                    metadata = self.create_student_metadata(student_data)
                    
                    texts.append(formatted_text)
                    metadata_list.append(metadata)
            
            if texts:
                return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
            else:
                print("❌ No valid student data found")
                return False
        
        except Exception as e:
            print(f"❌ Error processing unstructured Excel: {e}")
            return False

    def process_student_pdf(self, filename):
        """Process Student PDF with universal extraction"""
        try:
            doc = fitz.open(filename)
            all_text = ""
            for page in doc:
                all_text += page.get_text() + "\n"
            doc.close()
            
            # Split into potential student records
            student_records = self.split_into_student_records(all_text)
            
            texts = []
            metadata_list = []
            
            for record in student_records:
                student_data = self.extract_universal_student_data(record, 'pdf')
                
                if student_data['student_id']:
                    formatted_text = self.format_student_data(student_data)
                    metadata = self.create_student_metadata(student_data)
                    
                    texts.append(formatted_text)
                    metadata_list.append(metadata)
            
            if texts:
                return self.store_with_smart_hierarchy(texts, metadata_list, 'students')
            else:
                print("❌ No valid student data found")
                return False
        
        except Exception as e:
            print(f"❌ Error processing student PDF: {e}")
            return False

    def format_student_data(self, student_data):
        """Format extracted student data consistently"""
        return f"""
Student ID: {student_data.get('student_id', 'N/A')}
Full Name: {student_data.get('full_name', 'N/A')}
Surname: {student_data.get('surname', 'N/A')}
First Name: {student_data.get('first_name', 'N/A')}
Year: {student_data.get('year', 'N/A')}
Course: {student_data.get('course', 'N/A')}
Section: {student_data.get('section', 'N/A')}
Contact Number: {student_data.get('contact_number', 'N/A')}
Guardian Name: {student_data.get('guardian_name', 'N/A')}
Guardian Contact: {student_data.get('guardian_contact', 'N/A')}
""".strip()

    def create_student_metadata(self, student_data):
        """
        Create metadata from extracted student data, including the new 'searchable_name' field.
        """
        metadata = student_data.copy()
        
        # --- Create the "Super-Name" Field ---
        first_name = str(metadata.get('first_name', '')).lower()
        surname = str(metadata.get('surname', '')).lower()
        full_name = str(metadata.get('full_name', '')).lower()
        guardian_name = str(metadata.get('guardian_name', '')).lower()
        
        # Combine all possible names into one searchable string
        metadata['searchable_name'] = f"{first_name} {surname} {full_name} {guardian_name}"
        
        # --- Standardize other fields ---
        metadata['year_level'] = student_data.get('year')
        if metadata.get('course'):
            metadata['course'] = str(metadata['course']).upper()
        
        return {k: v for k, v in metadata.items() if v is not None}

    # ======================== COR PROCESSING ========================
    
    def process_cor_excel(self, filename):
        """Process COR Excel file with smart organization"""
        cor_info = self.extract_cor_excel_info(filename)
        formatted_text = self.format_cor_info(cor_info)
        
        # Create smart metadata
        metadata = {
            'course': cor_info['program_info']['Program'],
           'section': cor_info['program_info']['Section'],
           'year_level': cor_info['program_info']['Year Level'],
           'adviser': cor_info['program_info']['Adviser'],
           'data_type': 'cor_excel',
           'subject_codes': [course.get('Subject Code', '') for course in cor_info['schedule'] if course.get('Subject Code')]
       }
       
       # Smart department detection
        metadata['department'] = self.detect_department_from_course(metadata['course'])
        
        # Store with hierarchy
        collection_name = self.create_smart_collection_name('schedules', metadata)
        collection = self.client.get_or_create_collection(name=collection_name)
        self.store_with_smart_metadata(collection, [formatted_text], [metadata])
        self.collections[collection_name] = collection
        
        hierarchy_path = f"{metadata['department']} > {metadata['course']} > Year {metadata['year_level']} > Section {metadata['section']}"
        print(f"✅ Loaded COR schedule into: {collection_name}")
        print(f"   📁 Hierarchy: {hierarchy_path}")
        return True
   
    def process_cor_pdf(self, filename):
        """Process COR PDF file with smart organization"""
        cor_info = self.extract_cor_pdf_info(filename)
        if not cor_info:
            print("❌ Could not extract COR data from PDF")
            return False
        
        formatted_text = self.format_cor_info(cor_info)
        
        # Create smart metadata
        metadata = {
            'course': cor_info['program_info']['Program'],
            'section': cor_info['program_info']['Section'],
            'year_level': cor_info['program_info']['Year Level'],
            'adviser': cor_info['program_info']['Adviser'],
            'data_type': 'cor_pdf',
            'subject_codes': [course.get('Subject Code', '') for course in cor_info['schedule'] if course.get('Subject Code')]
        }
        
        # Smart department detection
        metadata['department'] = self.detect_department_from_course(metadata['course'])
        
        # Store with hierarchy
        collection_name = self.create_smart_collection_name('schedules', metadata)
        collection = self.client.get_or_create_collection(name=collection_name)
        self.store_with_smart_metadata(collection, [formatted_text], [metadata])
        self.collections[collection_name] = collection
        
        hierarchy_path = f"{metadata['department']} > {metadata['course']} > Year {metadata['year_level']} > Section {metadata['section']}"
        print(f"✅ Loaded COR schedule into: {collection_name}")
        print(f"   📁 Hierarchy: {hierarchy_path}")
        return True
   
   # ======================== FACULTY PROCESSING ========================
   
    def process_faculty_excel(self, filename):
        """Process Faculty Excel file with smart organization"""
        try:
            df = pd.read_excel(filename, header=None)
            
            # Extract faculty data from columns A and B (rows 1-45)
            faculty_data = {}
            current_section = ""
            
            for i in range(min(45, len(df))):  # Process up to row 45 or end of file
                field_name = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
                field_value = str(df.iloc[i, 1]).strip() if pd.notna(df.iloc[i, 1]) and str(df.iloc[i, 1]).strip() != 'nan' else ""
                
                # Skip empty rows
                if not field_name:
                    continue
                
                # Check if this is a section header (ALL CAPS)
                if field_name.isupper() and not field_value:
                    current_section = field_name
                    faculty_data[current_section] = {}
                else:
                    # Add field to current section or general data
                    if current_section:
                        faculty_data[current_section][field_name] = field_value
                    else:
                        faculty_data[field_name] = field_value
            
            # Format as text for ChromaDB storage
            formatted_text = self.format_faculty_excel_data(faculty_data)
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_excel')
            metadata['faculty_type'] = 'profile'
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(name=collection_name)
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            # Extract name for display
            faculty_name = faculty_data.get("PERSONAL INFORMATION", {}).get("Full Name", "Unknown Faculty")
            
            print(f"✅ Loaded faculty data into: {collection_name}")
            print(f"   Faculty: {faculty_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty Excel: {e}")
            return False
        
    def process_faculty_pdf(self, filename):
        """Process Faculty PDF file (Resume) with smart organization"""
        try:
            faculty_data = self.extract_faculty_pdf_data(filename)
            if not faculty_data:
                print("❌ Could not extract faculty data from PDF")
                return False
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(faculty_data, 'faculty_pdf')
            metadata['faculty_type'] = 'profile'
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(name=collection_name)
            self.store_with_smart_metadata(collection, [faculty_data], [metadata])
            self.collections[collection_name] = collection
            
            # Extract name from data for display
            lines = faculty_data.split('\n')
            faculty_name = lines[0] if lines else "Unknown Faculty"
            
            print(f"✅ Loaded faculty resume into: {collection_name}")
            print(f"   Faculty: {faculty_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty PDF: {e}")
            return False

    def process_faculty_schedule_excel(self, filename):
        """Process Faculty Schedule Excel file with smart organization"""
        try:
            df = pd.read_excel(filename, header=None)
            
            # Extract adviser name from first row
            adviser_name = ""
            for i in range(min(3, len(df))):
                for j in range(min(10, df.shape[1])):
                    cell_value = str(df.iloc[i, j]) if pd.notna(df.iloc[i, j]) else ""
                    if "ADVISER" in cell_value.upper():
                        # Look for the adviser name in nearby cells
                        if j + 1 < df.shape[1] and pd.notna(df.iloc[i, j + 1]):
                            adviser_name = str(df.iloc[i, j + 1])
                        break
            
            # Extract schedule data (simplified for now)
            schedule_data = []
            # ... (schedule extraction logic) ...
            
            # Format for ChromaDB
            formatted_text = self.format_faculty_schedule_data(adviser_name, schedule_data)
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_schedule_excel')
            metadata['faculty_type'] = 'teaching'
            metadata['adviser'] = adviser_name
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(name=collection_name)
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            print(f"✅ Loaded faculty schedule into: {collection_name}")
            print(f"   Adviser: {adviser_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty schedule Excel: {e}")
            return False

    def process_faculty_schedule_pdf(self, filename):
        """Process Faculty Schedule PDF file with smart organization"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Find adviser name
            adviser_name = "Unknown"
            for line in lines:
                if "NAME OF ADVISER:" in line:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            name_part = parts[1].strip()
                            if name_part:
                                adviser_name = name_part
                                break
            
            # Format the schedule data
            formatted_text = f"FACULTY CLASS SCHEDULE\nName of Adviser: {adviser_name}\n\n"
            formatted_text += full_text
            
            # Create smart metadata
            metadata = self.extract_smart_metadata(formatted_text, 'faculty_schedule_pdf')
            metadata['faculty_type'] = 'teaching'
            metadata['adviser'] = adviser_name
            
            # Store with hierarchy
            collection_name = self.create_smart_collection_name('faculty', metadata)
            collection = self.client.get_or_create_collection(name=collection_name)
            self.store_with_smart_metadata(collection, [formatted_text], [metadata])
            self.collections[collection_name] = collection
            
            print(f"✅ Loaded faculty schedule into: {collection_name}")
            print(f"   Adviser: {adviser_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing faculty schedule PDF: {e}")
            return False

   # ======================== FILE TYPE DETECTION ========================
   
    def is_cor_file(self, df):
        """Check if Excel is a COR file"""
        try:
            return (df.iloc[0, 0] == "Program:" and 
                    df.iloc[1, 0] == "Year Level:" and 
                    df.iloc[2, 0] == "Section:" and 
                    df.iloc[3, 0] == "Adviser:")
        except:
            return False
    
    def is_cor_pdf(self, filename):
        """Check if PDF is a COR file by looking for schedule keywords"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            # Look for COR-specific keywords
            cor_keywords = ["program:", "year level:", "section:", "adviser:", "subject code", "description", "units"]
            cor_count = sum(1 for keyword in cor_keywords if keyword in first_page)
            
            # Must have multiple COR-specific indicators and NOT be a class schedule
            return cor_count >= 4 and "class schedule" not in first_page
        except:
            return False

    def is_faculty_excel(self, df):
        """Check if Excel is a Faculty file"""
        try:
            # Check if first column contains faculty-specific headers
            first_col = df.iloc[:, 0].astype(str).str.upper()
            faculty_keywords = ["PERSONAL INFORMATION", "CONTACT INFORMATION", "OCCUPATIONAL INFORMATION"]
            return any(keyword in first_col.values for keyword in faculty_keywords)
        except:
            return False

    def is_faculty_pdf(self, filename):
        """Check if PDF is a Faculty file by looking for resume keywords"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            # Look for resume-specific keywords
            resume_keywords = ["professional profile", "education", "professional experience", "certifications", "email:", "phone:"]
            return any(keyword in first_page for keyword in resume_keywords)
        except:
            return False

    def is_faculty_schedule_excel(self, df):
        """Check for Faculty Schedule Excel files"""
        try:
            df_str = df.astype(str)
            first_few_rows = ' '.join(df_str.iloc[:10].values.flatten()).upper()
            
            faculty_schedule_indicators = [
                "NAME OF ADVISER",
                "CLASS SCHEDULE", 
                "FACULTY SCHEDULE",
                "TEACHING SCHEDULE"
            ]
            
            has_faculty_indicator = any(indicator in first_few_rows for indicator in faculty_schedule_indicators)
            has_day_layout = any(day in first_few_rows for day in ["MON", "TUE", "WED", "THU", "FRI"])
            
            # Should NOT have student data indicators
            student_indicators = ["STUDENT ID", "CONTACT NUMBER", "GUARDIAN"]
            has_student_indicator = any(indicator in first_few_rows for indicator in student_indicators)
            
            return has_faculty_indicator and has_day_layout and not has_student_indicator
            
        except:
            return False

    def is_faculty_schedule_pdf(self, filename):
        """Check if PDF is a Faculty Schedule file"""
        try:
            doc = fitz.open(filename)
            first_page = doc[0].get_text().lower()
            doc.close()
            
            schedule_keywords = ["class schedule", "name of adviser", "subject/s time"]
            faculty_schedule_indicators = ["mon tue wed thu fri", "subject/s", "time mon tue"]
            
            has_schedule_title = any(keyword in first_page for keyword in schedule_keywords)
            has_day_layout = any(indicator in first_page for indicator in faculty_schedule_indicators)
            
            return has_schedule_title and has_day_layout
        except:
            return False

    # ======================== HELPER METHODS ========================
    
    def extract_cor_excel_info(self, filename):
        """Extract COR information from Excel"""
        raw_df = pd.read_excel(filename, header=None)
        
        program_info = {
            'Program': raw_df.iloc[0, 1] if pd.notna(raw_df.iloc[0, 1]) else "",
            'Year Level': raw_df.iloc[1, 1] if pd.notna(raw_df.iloc[1, 1]) else "",
            'Section': raw_df.iloc[2, 1] if pd.notna(raw_df.iloc[2, 1]) else "",
            'Adviser': raw_df.iloc[3, 1] if pd.notna(raw_df.iloc[3, 1]) else ""
        }
        
        schedule_df = pd.read_excel(filename, header=4)
        schedule_data = []
        
        for _, row in schedule_df.iterrows():
            if pd.notna(row.iloc[0]) and not "Generated on:" in str(row.iloc[0]):
                schedule_data.append({
                    'Subject Code': row.iloc[0],
                    'Description': row.iloc[1],
                    'Type': row.iloc[2],
                    'Units': row.iloc[3],
                    'Day': row.iloc[4],
                    'Time Start': row.iloc[5],
                    'Time End': row.iloc[6],
                    'Room': row.iloc[7]
                })
        
        total_units = None
        for i in range(1, 5):
            try:
                if "Total Units:" in str(schedule_df.iloc[-i, 2]):
                    total_units = schedule_df.iloc[-i, 3]
                    break
            except:
                pass
        
        return {
            'program_info': program_info,
            'schedule': schedule_data,
            'total_units': total_units
        }
    
    def extract_cor_pdf_info(self, filename):
        """Extract COR information from PDF"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Extract program info
            program_info = {
                'Program': '',
                'Year Level': '',
                'Section': '',
                'Adviser': ''
            }
            
            for i, line in enumerate(lines):
                if line == 'Program:' and i + 1 < len(lines):
                    program_info['Program'] = lines[i + 1]
                elif line == 'Year Level:' and i + 1 < len(lines):
                    program_info['Year Level'] = lines[i + 1]
                elif line == 'Section:' and i + 1 < len(lines):
                    program_info['Section'] = lines[i + 1]
                elif line == 'Adviser:' and i + 1 < len(lines):
                    program_info['Adviser'] = lines[i + 1]
            
            # Extract schedule data
            schedule_data = []
            total_units = None
            
            # Find where the data starts after the header
            data_start = -1
            for i, line in enumerate(lines):
                if line == 'Room':  # This is the last header field
                    data_start = i + 1
                    break
            
            if data_start > 0:
                i = data_start
                while i + 7 < len(lines):
                    if lines[i] == 'Total Units':
                        total_units = lines[i + 1] if i + 1 < len(lines) else None
                        break
                    
                    if 'Generated on:' in lines[i]:
                        break
                    
                    try:
                        subject_entry = {
                            'Subject Code': lines[i],
                            'Description': lines[i + 1],
                            'Type': lines[i + 2],
                            'Units': lines[i + 3],
                            'Day': lines[i + 4],
                            'Time Start': lines[i + 5],
                            'Time End': lines[i + 6],
                            'Room': lines[i + 7]
                        }
                        
                        schedule_data.append(subject_entry)
                        i += 8
                        
                    except IndexError:
                        break
            
            return {
                'program_info': program_info,
                'schedule': schedule_data,
                'total_units': total_units
            }
            
        except Exception as e:
            print(f"❌ Error extracting COR PDF: {e}")
            return None

    def format_cor_info(self, cor_info):
        """Format COR information as text"""
        text = f"""
    Program: {cor_info['program_info']['Program']}
    Year Level: {cor_info['program_info']['Year Level']}
    Section: {cor_info['program_info']['Section']}
    Adviser: {cor_info['program_info']['Adviser']}
    Total Units: {cor_info['total_units']}

    Schedule:
    """
        for course in cor_info['schedule']:
            if course.get('Subject Code') and str(course['Subject Code']).lower() != 'nan':
                text += f"""
    - {course['Subject Code']} ({course.get('Type', 'N/A')}): {course.get('Description', 'N/A')}
    Day: {course.get('Day', 'N/A')}, Time: {course.get('Time Start', 'N/A')} to {course.get('Time End', 'N/A')}
    Room: {course.get('Room', 'N/A')}, Units: {course.get('Units', 'N/A')}
    """
        return text.strip()

    def extract_from_structured_text(self, lines):
        """Extract data from structured column:value format"""
        student_data = {
            'student_id': None, 'surname': None, 'first_name': None, 'full_name': None,
            'year': None, 'course': None, 'section': None, 'contact_number': None,
            'guardian_name': None, 'guardian_contact': None
        }
        
        # Map common column names to our fields
        field_mapping = {
            'STUDENT ID': 'student_id',
            'STUDENT_ID': 'student_id',
            'ID': 'student_id',
            'NAME': 'full_name',
            'STUDENT NAME': 'full_name',
            'FULL NAME': 'full_name',
            'SURNAME': 'surname',
            'FIRST NAME': 'first_name',
            'FIRSTNAME': 'first_name',
            'YEAR': 'year',
            'YEAR LEVEL': 'year',
            'COURSE': 'course',
            'PROGRAM': 'course',
            'SECTION': 'section',
            'CONTACT NUMBER': 'contact_number',
            'PHONE': 'contact_number',
            'MOBILE': 'contact_number',
            'GUARDIAN NAME': 'guardian_name',
            'GUARDIAN': 'guardian_name',
            'PARENT NAME': 'guardian_name',
            'GUARDIAN CONTACT': 'guardian_contact',
            "GUARDIAN'S CONTACT NUMBER": 'guardian_contact',
            'GUARDIAN CONTACT NUMBER': 'guardian_contact',
            'PARENT CONTACT': 'guardian_contact'
        }
        
        # Extract values from lines
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    column_name = parts[0].strip().upper()
                    value = parts[1].strip()
                    
                    # Skip header values and empty values
                    if (value and value != 'nan' and 
                        value.upper() not in ['SURNAME', 'FIRST NAME', 'GUARDIAN NAME', 'CONTACT NUMBER'] and
                        column_name in field_mapping):
                        
                        field = field_mapping[column_name]
                        student_data[field] = self.clean_extracted_value(value, field)
        
        # Post-process: if we have separate surname/first name, combine for full name
        if student_data['surname'] and student_data['first_name']:
            student_data['full_name'] = f"{student_data['surname']}, {student_data['first_name']}"
        elif student_data['full_name'] and not (student_data['surname'] and student_data['first_name']):
            # Split full name if we don't have separate fields
            student_data['surname'], student_data['first_name'] = self.split_full_name(student_data['full_name'])
        
        return student_data
    
    def extract_faculty_pdf_data(self, filename):
        """Extract faculty resume data from PDF"""
        try:
            doc = fitz.open(filename)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            # Clean and format the resume text
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Format as a structured resume
            formatted_text = "FACULTY RESUME\n\n"
            current_section = ""
            
            for line in lines:
                # Detect section headers
                if any(keyword in line.upper() for keyword in ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'CERTIFICATIONS', 'PROFILE']):
                    current_section = line
                    formatted_text += f"\n{line}\n" + "-" * len(line) + "\n"
                else:
                    formatted_text += f"{line}\n"
            
            return formatted_text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting faculty PDF: {e}")
            return None

    def format_faculty_excel_data(self, faculty_data):
        """Format faculty Excel data as text"""
        text = ""
        
        for section, data in faculty_data.items():
            if isinstance(data, dict):
                text += f"\n{section}:\n"
                for field, value in data.items():
                    if value:  # Only include non-empty values
                        text += f"  {field}: {value}\n"
            else:
                if data:  # Only include non-empty values
                    text += f"{section}: {data}\n"
        
        return text.strip()

    def format_faculty_schedule_data(self, adviser_name, schedule_data):
        """Format faculty schedule data as text"""
        text = f"FACULTY CLASS SCHEDULE\nName of Adviser: {adviser_name}\n\n"
        
        if schedule_data:
            text += "WEEKLY SCHEDULE:\n"
            for entry in schedule_data:
                text += f"Day: {entry.get('Day', 'N/A')}\n"
                text += f"Time: {entry.get('Time', 'N/A')}\n"
                text += f"Subject: {entry.get('Subject', 'N/A')}\n"
                text += f"Class: {entry.get('Class', 'N/A')}\n"
                text += "-" * 30 + "\n"
        
        return text.strip()

    # ======================== FILE PROCESSING CONTROLLER ========================
    
    def process_file(self, filename):
        """Smart file processing controller"""
        ext = os.path.splitext(filename)[1].lower()
        
        try:
            if ext == ".xlsx":
                df_check = pd.read_excel(filename, header=None)
                if self.is_cor_file(df_check):
                    return self.process_cor_excel(filename)
                elif self.is_faculty_excel(df_check):
                    return self.process_faculty_excel(filename)
                elif self.is_faculty_schedule_excel(df_check):
                    return self.process_faculty_schedule_excel(filename)
                else:
                    return self.process_student_excel(filename)
            elif ext == ".pdf":
                if self.is_cor_pdf(filename):
                    return self.process_cor_pdf(filename)
                elif self.is_faculty_pdf(filename):
                    return self.process_faculty_pdf(filename)
                elif self.is_faculty_schedule_pdf(filename):
                    return self.process_faculty_schedule_pdf(filename)
                else:
                    return self.process_student_pdf(filename)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
        return False

    def load_new_data(self):
        """Load new data from files"""
        files = self.list_available_files()
        if not files:
            return False
            
        try:
            choice = int(input("\n🔢 Enter file number to load: ").strip())
            if 1 <= choice <= len(files):
                filename = files[choice - 1]
                success = self.process_file(filename)
                if success:
                    self.data_loaded = True
                    print("✅ Data loaded successfully!")
                return success
        except ValueError:
            print("❌ Invalid input.")
        return False

    # ======================== SEARCH & QUERY ========================
    
    def search_all_collections(self, query, max_results=30):
        """Search across all loaded collections"""
        all_results = []
        
        for name, collection in self.collections.items():
            try:
                results = collection.query(query_texts=[query], n_results=max_results)
                if results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        metadata = results["metadatas"][0][i] if results["metadatas"][0] else {}
                        collection_type = self.get_collection_type(name)
                        all_results.append({
                            "source": collection_type, 
                            "content": doc,
                            "metadata": metadata,
                            "hierarchy": f"{metadata.get('department', 'Unknown')} > {metadata.get('course', 'Unknown')} > {metadata.get('section', 'Unknown')}"
                        })
            except Exception as e:
                print(f"⚠️ Error searching {name}: {e}")
        
        return all_results

    def smart_search_with_filters(self, user_query, name_query_term, where_clause, max_results=50):
        if name_query_term:
            if self.debug_mode:
                print(f"⚙️ Performing broad search for name: '{name_query_term}', then filtering...")
            
            broad_results = self._perform_query(user_query, where_clause, max_results)
            
            final_results = []
            for res in broad_results:
                searchable_name = res.get('metadata', {}).get('searchable_name', '')
                if name_query_term.lower() in searchable_name:
                    final_results.append(res)
            
            return final_results
        else:
            if where_clause:
                if self.debug_mode:
                    print(f"⚙️ Applying precise database filters: {where_clause}")
            else:
                if self.debug_mode:
                    print("⚙️ No filters provided. Performing broad semantic search...")
            
            return self._perform_query(user_query, where_clause, max_results)

    def _perform_query(self, user_query, where_clause, max_results):
        all_results = []
        for collection in self.client.list_collections():
            if not collection.name.startswith("students"):
                continue
            try:
                query_params = {
                    'query_texts': [user_query],
                    'n_results': max_results
                }
                if where_clause:
                    query_params['where'] = where_clause
                
                results_dict = collection.query(**query_params)
                
                for i in range(len(results_dict['ids'][0])):
                    all_results.append({
                        'id': results_dict['ids'][0][i],
                        'content': results_dict['documents'][0][i],
                        'metadata': results_dict['metadatas'][0][i],
                        'distance': results_dict['distances'][0][i],
                        'source': collection.name
                    })
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Could not query {collection.name} with filters: {e}")
                pass
        
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:max_results]
    
    
    
    
    
    ### NEW: SETTINGS MANAGEMENT ###
    def manage_settings(self):
        """A new menu to manage system settings."""
        while True:
            print("\n⚙️ SYSTEM SETTINGS")
            print("-" * 20)
            print(f"1. Debug Mode  : {'ON' if self.debug_mode else 'OFF'}")
            print(f"2. API Mode    : {self.api_mode.upper()}")
            print("3. Back to Main Menu")
            
            choice = input("\nChoose an option to toggle or change: ").strip()
            
            if choice == '1':
                self.debug_mode = not self.debug_mode
                print(f"✅ Debug mode is now {'ON' if self.debug_mode else 'OFF'}.")
            elif choice == '2':
                self.api_mode = 'offline' if self.api_mode == 'online' else 'online'
                print(f"✅ API mode is now set to '{self.api_mode.upper()}'.")
            elif choice == '3':
                break
            else:
                print("❌ Invalid choice.")
                
                

    def exact_match_search(self, query):
        """Perform exact text matching across all collections"""
        matches = []
        for name, collection in self.collections.items():
            try:
                # Get all documents from collection
                all_docs = collection.get()
                for doc in all_docs["documents"]:
                    if query.lower() in doc.lower():
                        collection_type = self.get_collection_type(name)
                        matches.append({"source": collection_type, "content": doc})
            except Exception as e:
                print(f"⚠️ Error in exact search for {name}: {e}")
        return matches

    def search_specific_collection(self, collection_name, query, max_results=5):
        """Search in a specific collection"""
        if collection_name not in self.collections:
            print(f"❌ Collection '{collection_name}' not found")
            return []
        
        try:
            collection = self.collections[collection_name]
            results = collection.query(query_texts=[query], n_results=max_results)
            return results["documents"][0] if results["documents"][0] else []
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []

    # ======================== COLLECTION MANAGEMENT ========================
    
    def delete_collection(self, collection_name):
        """Delete a specific collection"""
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            print(f"✅ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection {collection_name}: {e}")
            return False

    def delete_all_collections(self):
        """Delete all collections"""
        try:
            existing_collections = self.client.list_collections()
            deleted_count = 0
            
            for collection in existing_collections:
                try:
                    self.client.delete_collection(name=collection.name)
                    if collection.name in self.collections:
                        del self.collections[collection.name]
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Could not delete {collection.name}: {e}")
            
            print(f"✅ Deleted {deleted_count} collections")
            self.data_loaded = False
            return True
        except Exception as e:
            print(f"❌ Error deleting collections: {e}")
            return False

    def manage_collections(self):
        """Collection management interface"""
        existing = self.check_existing_data()
        
        if not existing:
            print("❌ No collections found to manage.")
            return
        
        print("\n📚 COLLECTION MANAGEMENT")
        print("=" * 50)
        print("1. 🗑️  Delete specific collection")
        print("2. 🗑️  Delete all collections")
        print("3. 📋 Show collection details")
        print("4. ↩️  Back to main menu")
        
        try:
            choice = input("\n💡 Choose an option (1-4): ").strip()
            
            if choice == "1":
                self.delete_specific_collection_menu()
            elif choice == "2":
                self.delete_all_collections_menu()
            elif choice == "3":
                self.show_all_collections()
            elif choice == "4":
                return
            else:
                print("❌ Invalid choice.")
        except KeyboardInterrupt:
            print("\n↩️ Returning to main menu...")

    def delete_specific_collection_menu(self):
        """Menu for deleting specific collections"""
        existing = self.client.list_collections()
        
        if not existing:
            print("❌ No collections found.")
            return
        
        print(f"\n📚 Available Collections:")
        for i, collection in enumerate(existing, 1):
            count = collection.count()
            collection_type = self.get_collection_type(collection.name)
            print(f"  {i}. {collection_type} ({count} records)")
        
        try:
            choice = int(input("\n🔢 Enter collection number to delete (0 to cancel): ").strip())
            
            if choice == 0:
                print("❌ Cancelled.")
                return
            
            if 1 <= choice <= len(existing):
                collection_to_delete = existing[choice - 1]
                collection_type = self.get_collection_type(collection_to_delete.name)
                
                # Confirm deletion
                confirm = input(f"\n⚠️  Are you sure you want to delete '{collection_type}'? (yes/no): ").strip().lower()
                
                if confirm in ['yes', 'y']:
                    if self.delete_collection(collection_to_delete.name):
                        print(f"✅ Successfully deleted '{collection_type}'")
                        
                        # Ask if they want to reload this type of data
                        reload = input(f"\n🔄 Do you want to reload {collection_type} from files? (yes/no): ").strip().lower()
                        if reload in ['yes', 'y']:
                            self.reload_specific_data_type(collection_to_delete.name)
                    else:
                        print(f"❌ Failed to delete '{collection_type}'")
                else:
                    print("❌ Deletion cancelled.")
            else:
                print("❌ Invalid selection.")
                
        except ValueError:
            print("❌ Invalid input.")

    def delete_all_collections_menu(self):
        """Menu for deleting all collections with confirmation"""
        existing = self.client.list_collections()
        
        if not existing:
            print("❌ No collections found.")
            return
        
        print(f"\n⚠️  WARNING: This will delete ALL {len(existing)} collections:")
        for collection in existing:
            collection_type = self.get_collection_type(collection.name)
            count = collection.count()
            print(f"   • {collection_type} ({count} records)")
        
        confirm = input(f"\n⚠️  Are you sure you want to delete ALL collections? Type 'DELETE ALL' to confirm: ").strip()
        
        if confirm == 'DELETE ALL':
            if self.delete_all_collections():
                print("✅ All collections deleted successfully!")
                
                # Ask if they want to reload data
                reload = input(f"\n🔄 Do you want to load fresh data now? (yes/no): ").strip().lower()
                if reload in ['yes', 'y']:
                    self.load_new_data()
            else:
                print("❌ Failed to delete all collections.")
        else:
            print("❌ Deletion cancelled.")

    def reload_specific_data_type(self, collection_name):
        """Reload a specific type of data"""
        # Map collection names to file types
        file_type_map = {
            "students_excel": "student Excel",
            "students_pdf": "student PDF", 
            "schedules_excel": "COR Excel",
            "schedules_pdf": "COR PDF",
            "faculty_excel": "faculty Excel",
            "faculty_pdf": "faculty PDF",
            "faculty_schedules_excel": "faculty schedule Excel",
            "faculty_schedules_pdf": "faculty schedule PDF"
        }
        
        data_type = file_type_map.get(collection_name, "unknown")
        print(f"\n🔍 Looking for {data_type} files...")
        
        files = self.list_available_files()
        if not files:
            print("❌ No files available to load.")
            return
        
        # Filter files by type if possible
        relevant_files = []
        for i, file in enumerate(files):
            file_type = self.detect_file_type(file)
            if data_type.lower() in file_type.lower():
                relevant_files.append((i + 1, file, file_type))
        
        if relevant_files:
            print(f"\n📁 Found {len(relevant_files)} relevant files:")
            for idx, file, ftype in relevant_files:
                print(f"  {idx}. {file} - {ftype}")
            
            try:
                choice = int(input(f"\n🔢 Enter file number to load: ").strip())
                if any(choice == idx for idx, _, _ in relevant_files):
                    filename = files[choice - 1]
                    self.process_file(filename)
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Invalid input.")
        else:
            print(f"❌ No {data_type} files found. Showing all available files:")
            choice = int(input("\n🔢 Enter file number to load: ").strip())
            if 1 <= choice <= len(files):
                filename = files[choice - 1]
                self.process_file(filename)

    # ======================== USER INTERFACE ========================
    
    def show_search_options(self):
        """Display search options to user"""
        print("\n🔍 SEARCH OPTIONS:")
        print("1. 🔎 Smart Search (AI-powered similarity)")
        print("2. 📝 Exact Match Search")
        print("3. 📊 Browse by Collection")
        print("4. 📂 Load More Data")
        print("5. 📋 Show All Collections")
        print("6. 🗑️  Manage Collections")
        print("7. ❌ Exit")
        
        if self.collections:
            print(f"\n📚 Loaded Collections:")
            for name in self.collections.keys():
                count = self.collections[name].count()
                collection_type = self.get_collection_type(name)
                print(f"   • {collection_type} ({count} records)")

    def smart_search(self):
        """Smart AI-powered search interface"""
        query = input("\n🧠 Enter your search query: ").strip()
        if not query:
            return
        
        print(f"\n🔍 Searching for: '{query}'")
        results = self.search_all_collections(query)
        
        if results:
            print(f"\n✅ Found {len(results)} relevant results:")
            for i, result in enumerate(results, 1):
                print(f"\n📄 Result {i} (from {result['source']}):")
                print(f"📁 {result.get('hierarchy', 'N/A')}")
                print("-" * 60)
                print(result['content'])
        else:
            print("❌ No relevant results found.")

    def exact_search(self):
        """Exact text matching search"""
        query = input("\n📝 Enter exact text to find: ").strip()
        if not query:
            return
        
        print(f"\n🔍 Searching for exact matches: '{query}'")
        matches = self.exact_match_search(query)
        
        if matches:
            print(f"\n✅ Found {len(matches)} exact matches:")
            for i, match in enumerate(matches, 1):
                print(f"\n📄 Match {i} (from {match['source']}):")
                print("-" * 60)
                print(match['content'])
        else:
            print("❌ No exact matches found.")

    def browse_collections(self):
        """Browse data by collection"""
        if not self.collections:
            print("❌ No collections available.")
            return
        
        print(f"\n📚 Available Collections:")
        collection_list = list(self.collections.keys())
        for i, name in enumerate(collection_list, 1):
            count = self.collections[name].count()
            collection_type = self.get_collection_type(name)
            print(f"  {i}. {collection_type} ({count} records)")
        
        try:
            choice = int(input("\n🔢 Choose collection number: ").strip())
            if 1 <= choice <= len(collection_list):
                collection_name = collection_list[choice - 1]
                collection_type = self.get_collection_type(collection_name)
                query = input(f"\n🔍 Search in '{collection_type}': ").strip()
                
                if query:
                    results = self.search_specific_collection(collection_name, query)
                    if results:
                        print(f"\n✅ Found {len(results)} results in {collection_type}:")
                        for i, result in enumerate(results, 1):
                            print(f"\n📄 Result {i}:")
                            print("-" * 60)
                            print(result)
                    else:
                        print("❌ No results found.")
        except ValueError:
            print("❌ Invalid input.")

    def show_all_collections(self):
        """Show detailed info about all collections"""
        if not self.collections:
            print("❌ No collections loaded.")
            return
        
        print(f"\n📊 COLLECTION DETAILS:")
        print("=" * 60)
        
        for name, collection in self.collections.items():
            collection_type = self.get_collection_type(name)
            count = collection.count()
            print(f"\n📁 {collection_type}")
            print(f"   Collection ID: {name}")
            print(f"   Records: {count}")
            
            # Show sample data
            try:
                sample = collection.get(limit=1)
                if sample["documents"]:
                    print(f"   Sample data:")
                    sample_text = sample["documents"][0][:200] + "..." if len(sample["documents"][0]) > 200 else sample["documents"][0]
                    print(f"   {sample_text}")
            except:
                pass
            print("-" * 40)

    def run_query_interface(self):
        """Main query interface, now with a 'settings' option."""
        
        if not self.data_loaded:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*70)
        print("🎯 SMART STUDENT DATA SYSTEM - READY!")
        print("GAMITIN NIYO YUNG NUMBER 7 to off debugging and switch to offline")
        print("="*70)
        
        while True:
            # Your original menu options are preserved
            print("\n🔍 SEARCH OPTIONS:")
            print("1. 🔎 Smart Search")
            print("2. 📝 Exact Match Search")
            print("3. 📊 Browse by Collection")
            print("4. 📂 Load More Data")
            print("5. 📋 Show All Collections")
            print("6. 🗑️ Manage Collections")
            print("7. ⚙️ Settings") 
            print("8. ❌ Exit")
            print("\nTYPE 'test' to enter AI test mode (DITO KAYO MAG QUERY)")

            choice = input("\n💡 Choose an option: ").strip().lower()

            if choice == "1": self.smart_search()
            elif choice == "2": self.exact_search()
            elif choice == "3": self.browse_collections()
            elif choice == "4": self.load_new_data()
            elif choice == "5": self.show_all_collections()
            elif choice == "6": self.manage_collections()
            elif choice == "7": self.manage_settings() # New option
            elif choice == "8":
                print("👋 Goodbye!")
                break
            elif choice == "test":
                print("\n🧪 RAG Test Mode")
                while True:
                    query = input("\nAsk a question (type 'exit' to quit): ").strip()
                    if query.lower() == "exit":
                        break
                    answer = rag_query_pipeline(self, query)
                    print("\n🤖 Response:\n")
                    print(answer)
            else:
                print("❌ Invalid choice.")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 SMART STUDENT DATA SYSTEM")
    print("="*60)
    system = SmartStudentDataSystem()
    if system.quick_setup():
        system.run_query_interface()
    else:
        print("❌ Could not load any data. Please check your files and try again.")