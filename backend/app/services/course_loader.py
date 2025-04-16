import toml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from configs.config import Config
import time

logger = logging.getLogger(__name__)

class CourseLoader:
    def __init__(self):
        self._courses_cache = None
        self._last_loaded = None

    async def load_all_courses(self) -> Dict[str, Dict[str, Any]]:
        """Load all courses from TOML files with caching"""
        try:
            # Check if cache is still valid
            if self._courses_cache and self._cache_is_valid():
                return self._courses_cache
                
            courses = {}
            for course_file in Config.COURSES_DIR.glob("*.toml"):
                try:
                    file_mtime = course_file.stat().st_mtime
                    with open(course_file, 'r', encoding='utf-8') as f:
                        data = toml.load(f)
                        if 'course' not in data:
                            logger.warning(f"Invalid course file format in {course_file}")
                            continue
                            
                        course = self._validate_course(data['course'])
                        topics = self._process_topics(data.get('topics', []))
                        
                        courses[course['id']] = {
                            **course,
                            'topics': topics,
                            'last_modified': file_mtime
                        }
                except Exception as e:
                    logger.error(f"Error loading course file {course_file}: {e}")
                    continue
            
            self._courses_cache = courses
            self._last_loaded = time.time()
            return courses
            
        except Exception as e:
            logger.error(f"Failed to load courses: {e}")
            return {}

    def _cache_is_valid(self) -> bool:
        """Check if cached courses are still valid by checking file timestamps"""
        if not self._last_loaded:
            return False
            
        for course_id, course_data in self._courses_cache.items():
            if course_data.get('last_modified', 0) > self._last_loaded:
                return False
        return True

    def _validate_course(self, course_data: Dict) -> Dict:
        """Validate course structure and set defaults"""
        required_fields = ['id', 'title', 'description']
        if not all(field in course_data for field in required_fields):
            raise ValueError("Course missing required fields")
            
        return {
            'id': course_data['id'],
            'title': course_data['title'],
            'description': course_data.get('description', ''),
            'language': course_data.get('language', 'python')
        }

    def _process_topics(self, topics: List[Dict]) -> List[Dict]:
        """Process and validate topics structure"""
        validated_topics = []
        for topic in topics:
            try:
                if 'id' not in topic or 'title' not in topic:
                    logger.warning("Topic missing required fields, skipping")
                    continue
                    
                validated_topic = {
                    'id': topic['id'],
                    'title': topic['title'],
                    'description': topic.get('description', ''),
                    'problems': self._process_problems(topic.get('problems', []))
                }
                validated_topics.append(validated_topic)
            except Exception as e:
                logger.error(f"Error processing topic: {e}")
                continue
                
        return validated_topics

    def _process_problems(self, problems: List[Dict]) -> List[Dict]:
        """Process and validate problems structure"""
        validated_problems = []
        for problem in problems:
            try:
                if 'id' not in problem or 'title' not in problem:
                    logger.warning("Problem missing required fields, skipping")
                    continue
                    
                validated_problem = {
                    'id': problem['id'],
                    'title': problem['title'],
                    'description': problem.get('description', ''),
                    'complexity': problem.get('complexity', 'medium'),
                    'starter_code': problem.get('starter_code', ''),
                    'visible_test_cases': problem.get('visible_test_cases', []),
                    'hidden_test_cases': problem.get('hidden_test_cases', [])
                }
                
                # Validate test cases
                self._validate_test_cases(validated_problem['visible_test_cases'])
                self._validate_test_cases(validated_problem['hidden_test_cases'])
                
                validated_problems.append(validated_problem)
            except Exception as e:
                logger.error(f"Error processing problem: {e}")
                continue
                
        return validated_problems

    def _validate_test_cases(self, test_cases: List[Dict]):
        """Flexible test case validation that handles both 'output' and 'expected_output'"""
        if not isinstance(test_cases, list):
            raise ValueError("Test cases must be a list")
        
        for case in test_cases:
            if not isinstance(case, dict):
                raise ValueError("Each test case must be a dictionary")
            
            # Handle both 'output' and 'expected_output'
            if 'expected_output' in case:
                case['output'] = case['expected_output']
            elif 'output' not in case:
                raise ValueError("Test case missing required output field")
            
            # Set default empty string if input is missing
            if 'input' not in case:
                case['input'] = ""
            
            # Ensure output is string
            if not isinstance(case['output'], str):
                case['output'] = str(case['output'])

    async def get_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get a single course by ID"""
        try:
            courses = await self.load_all_courses()
            return courses.get(course_id)
        except Exception as e:
            logger.error(f"Error getting course {course_id}: {e}")
            return None

    async def get_topic(self, course_id: str, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get a topic from a course"""
        try:
            course = await self.get_course(course_id)
            if not course:
                return None
                
            return next(
                (t for t in course.get('topics', []) if t['id'] == topic_id),
                None
            )
        except Exception as e:
            logger.error(f"Error getting topic {topic_id}: {e}")
            return None

    async def get_problem(self, course_id: str, topic_id: str, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a problem from a topic"""
        try:
            topic = await self.get_topic(course_id, topic_id)
            if not topic:
                return None
                
            return next(
                (p for p in topic.get('problems', []) if p['id'] == problem_id),
                None
            )
        except Exception as e:
            logger.error(f"Error getting problem {problem_id}: {e}")
            return None