"""
Self-Modifying Code Module for AlphaVox

This module enables AlphaVox to modify its own code based on learning and adaptation.
It includes safety mechanisms to prevent catastrophic changes and maintains
backups of all modified files.
"""
import os
import sys
import ast
import logging
import inspect
import difflib
import importlib
import time
import json
import shutil
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_modifying_code")

# Check if Anthropic API key is available
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

class SafetyError(Exception):
    """Exception raised for safety check failures"""
    pass

class CodeModification:
    """Represents a code modification to be applied"""
    
    def __init__(self, 
                 file_path: str, 
                 original_code: str, 
                 modified_code: str, 
                 description: str,
                 modification_type: str,
                 confidence: float):
        self.file_path = file_path
        self.original_code = original_code
        self.modified_code = modified_code
        self.description = description
        self.modification_type = modification_type  # 'bugfix', 'optimization', 'feature'
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()
        self.applied = False
        self.result = None
    
    def get_diff(self) -> str:
        """Get a unified diff of the changes"""
        orig_lines = self.original_code.splitlines(keepends=True)
        modified_lines = self.modified_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            orig_lines, modified_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            n=3
        )
        
        return ''.join(diff)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "file_path": self.file_path,
            "description": self.description,
            "modification_type": self.modification_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "applied": self.applied,
            "result": self.result,
            "diff": self.get_diff()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], original_code: str, modified_code: str) -> 'CodeModification':
        """Create from dictionary"""
        modification = cls(
            file_path=data["file_path"],
            original_code=original_code,
            modified_code=modified_code,
            description=data["description"],
            modification_type=data["modification_type"],
            confidence=data["confidence"]
        )
        modification.timestamp = data["timestamp"]
        modification.applied = data["applied"]
        modification.result = data["result"]
        return modification

class CodeModifier:
    """Handles safe modification of code files"""
    
    def __init__(self, backup_dir: str = "data/backups"):
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Track modifications
        self.modifications = []
        self.load_modifications()
        
        # Safety thresholds
        self.min_confidence = 0.8  # Minimum confidence required for autonomous changes
        self.max_lines_changed = 100  # Maximum lines that can be changed at once
        self.safe_files = set()  # Files that can be modified
        
        # Initialize safe files list
        self._initialize_safe_files()
    
    def _initialize_safe_files(self):
        """Initialize the list of files that are safe to modify"""
        # Basic system files should never be modified
        unsafe_patterns = [
            "main.py",  # Main application entry
            ".git",     # Git files
            "db.py",    # Database configuration
            "pyproject.toml",  # Project configuration
            "requirements.txt", # Dependencies
            "Pipfile",  # More dependencies
            "setup.py", # Installation script
            "self_modifying_code.py"  # This file (never modify self)
        ]
        
        # All other Python files in the project are potentially modifiable
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    
                    # Check against unsafe patterns
                    if not any(unsafe in file_path for unsafe in unsafe_patterns):
                        # Remove leading "./" if present
                        if file_path.startswith("./"):
                            file_path = file_path[2:]
                        self.safe_files.add(file_path)
        
        logger.info(f"Initialized {len(self.safe_files)} safe files for modification")
    
    def load_modifications(self):
        """Load history of modifications from storage"""
        history_file = os.path.join(self.backup_dir, "modification_history.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as file:
                    history = json.load(file)
                
                # We don't load the actual modifications here - just the metadata
                # Actual code diffs would be too large to keep in memory
                self.modifications = history
                logger.info(f"Loaded {len(history)} previous modifications")
            except json.JSONDecodeError:
                logger.warning("Failed to load modification history, starting fresh")
                self.modifications = []
        else:
            self.modifications = []
    
    def save_modifications(self):
        """Save modification history to storage"""
        history_file = os.path.join(self.backup_dir, "modification_history.json")
        
        with open(history_file, 'w') as file:
            json.dump(self.modifications, file, indent=2)
    
    def create_backup(self, file_path: str) -> str:
        """Create a backup of a file before modification"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path)
        backup_name = f"{filename}.{timestamp}.bak"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path
    
    def check_syntax(self, code: str) -> bool:
        """Check if modified code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in proposed code: {str(e)}")
            return False
    
    def apply_modification(self, modification: CodeModification) -> bool:
        """Apply a code modification if it passes safety checks"""
        file_path = modification.file_path
        
        try:
            # Run safety checks
            self._run_safety_checks(modification)
            
            # Create backup
            backup_path = self.create_backup(file_path)
            
            # Write the modified code
            with open(file_path, 'w') as file:
                file.write(modification.modified_code)
            
            # Record successful application
            modification.applied = True
            modification.result = "success"
            
            # Update the modification history
            self.modifications.append(modification.to_dict())
            self.save_modifications()
            
            logger.info(f"Successfully applied modification to {file_path}")
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to apply modification: {str(e)}"
            logger.error(error_msg)
            
            # If the file was modified, restore from backup
            if getattr(e, 'restore_backup', False) and 'backup_path' in locals():
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Restored backup from {backup_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {str(restore_error)}")
            
            # Record failed application
            modification.applied = False
            modification.result = error_msg
            
            # Still record the attempt
            self.modifications.append(modification.to_dict())
            self.save_modifications()
            
            return False
    
    def _run_safety_checks(self, modification: CodeModification):
        """Run safety checks before applying a modification"""
        file_path = modification.file_path
        
        # Check if file is in the safe list
        if file_path not in self.safe_files:
            raise SafetyError(f"File {file_path} is not in the list of safely modifiable files")
        
        # Check confidence threshold
        if modification.confidence < self.min_confidence:
            raise SafetyError(f"Confidence {modification.confidence} is below the threshold {self.min_confidence}")
        
        # Check syntax
        if not self.check_syntax(modification.modified_code):
            raise SafetyError("Modified code contains syntax errors")
        
        # Check number of changed lines
        orig_lines = modification.original_code.splitlines()
        modified_lines = modification.modified_code.splitlines()
        
        # Simple diff to count changed lines
        changes = 0
        for i, (orig, modified) in enumerate(zip(orig_lines, modified_lines)):
            if orig != modified:
                changes += 1
        
        # Also count lines added or removed
        changes += abs(len(orig_lines) - len(modified_lines))
        
        if changes > self.max_lines_changed:
            raise SafetyError(f"Too many lines changed: {changes} (max: {self.max_lines_changed})")

class AICodeGenerator:
    """Uses AI to generate code improvements"""
    
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.api_endpoint = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"  # the newest Anthropic model is "claude-sonnet-4-20250514" which was released October 22, 2024
    
    def generate_code_improvement(self, file_path: str, code: str, issue_description: str) -> Tuple[str, str, float]:
        """
        Generate improved code using AI
        
        Args:
            file_path: Path to the file being modified
            code: Original code
            issue_description: Description of the issue to fix
            
        Returns:
            tuple: (improved_code, explanation, confidence)
        """
        if not self.api_key:
            logger.warning("No Anthropic API key available, using fallback code generation")
            return self._fallback_code_generation(code, issue_description)
        
        # Prepare the prompt
        prompt = f"""You are an expert Python developer helping to improve code for the AlphaVox AI system.
        
You need to fix or improve the following code from {file_path}:

```python
{code}
```

The issue to address is: {issue_description}

Please provide:
1. An improved version of the entire file that fixes the issue
2. A clear explanation of what you changed and why
3. A confidence score (0.0-1.0) indicating how confident you are that your changes are correct

Follow these guidelines:
- Make minimal changes necessary to fix the issue
- Maintain the same overall structure and function signatures
- Only fix the specified issue
- Do not introduce new features unless explicitly requested
- Ensure all imports are preserved
- Be especially careful with database models and API endpoints
"""

        try:
            # Make the API request
            headers = {
                "x-api-key": self.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("content", [{}])[0].get("text", "")
            
            # Extract the improved code, explanation and confidence
            improved_code = self._extract_code(content)
            explanation = self._extract_explanation(content)
            confidence = self._extract_confidence(content)
            
            if not improved_code:
                logger.warning("AI did not generate valid code")
                return self._fallback_code_generation(code, issue_description)
            
            return improved_code, explanation, confidence
            
        except Exception as e:
            logger.error(f"Error generating code improvement: {str(e)}")
            return self._fallback_code_generation(code, issue_description)
    
    def _extract_code(self, content: str) -> str:
        """Extract code from API response"""
        code_blocks = []
        in_code_block = False
        current_block = []
        
        for line in content.split('\n'):
            if line.startswith('```python'):
                in_code_block = True
            elif line.startswith('```') and in_code_block:
                in_code_block = False
                code_blocks.append('\n'.join(current_block))
                current_block = []
            elif in_code_block:
                current_block.append(line)
        
        # If there are code blocks, return the largest one
        if code_blocks:
            return max(code_blocks, key=len)
        return ""
    
    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from API response"""
        if "explanation" in content.lower():
            parts = content.lower().split("explanation")
            if len(parts) > 1:
                explanation_section = parts[1].split("confidence")[0]
                return explanation_section.strip()
        
        # Fallback: just return everything after the code
        code_end = content.rfind('```')
        if code_end > 0:
            return content[code_end+3:].strip()
        
        return "No explanation provided"
    
    def _extract_confidence(self, content: str) -> float:
        """Extract confidence score from API response"""
        # Look for patterns like "Confidence: 0.85" or "confidence score: 0.9"
        import re
        confidence_patterns = [
            r"confidence[:\s]+(\d+\.\d+)",
            r"confidence score[:\s]+(\d+\.\d+)",
            r"confidence level[:\s]+(\d+\.\d+)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        # Default confidence
        return 0.7
    
    def _fallback_code_generation(self, code: str, issue_description: str) -> Tuple[str, str, float]:
        """
        Fallback method when AI generation is not available
        This implements basic issue fixes based on common patterns
        """
        modified_code = code
        explanation = "Automatic fallback code generation"
        confidence = 0.6
        
        # Look for simple issues to fix
        if "circular import" in issue_description.lower():
            # Fix circular imports by using deferred imports
            if "import" in code and "from" in code:
                lines = code.splitlines()
                imports = []
                for i, line in enumerate(lines):
                    if line.startswith('from') or line.startswith('import'):
                        imports.append((i, line))
                
                # Move some imports inside functions
                for i, import_line in reversed(imports):
                    # Check if this import might be causing issues
                    if any(mod in import_line for mod in ['app', 'models', 'db']):
                        # Find the first function definition
                        for j, line in enumerate(lines):
                            if line.startswith('def '):
                                # Add the import at the start of the function
                                func_body_indent = len(line) - len(line.lstrip()) + 4
                                indented_import = ' ' * func_body_indent + import_line
                                lines.insert(j + 1, indented_import)
                                
                                # Remove the original import
                                lines.pop(i)
                                
                                explanation = f"Moved import '{import_line.strip()}' inside the first function to prevent circular imports"
                                modified_code = '\n'.join(lines)
                                confidence = 0.7
                                break
        
        elif "unused import" in issue_description.lower():
            # Remove unused imports
            lines = code.splitlines()
            removed_imports = []
            
            for i, line in enumerate(lines):
                if (line.startswith('import') or line.startswith('from')) and '#' not in line:
                    # This is a simple heuristic and might remove needed imports
                    module = line.split(' ')[1].split('.')[0] if 'import' in line else ""
                    if module and module not in code.replace(line, ""):
                        removed_imports.append(line.strip())
                        lines.pop(i)
            
            if removed_imports:
                explanation = f"Removed unused imports: {', '.join(removed_imports)}"
                modified_code = '\n'.join(lines)
                confidence = 0.7
        
        elif "undefined variable" in issue_description.lower():
            # Try to fix undefined variables by adding initializations
            undefined_var_match = re.search(r"undefined\s+variable\s+['\"](.*?)['\"]", issue_description.lower())
            if undefined_var_match:
                var_name = undefined_var_match.group(1)
                # Add initialization at the top of the file
                lines = code.splitlines()
                lines.insert(1, f"{var_name} = None  # Auto-initialized by self-modifying code")
                modified_code = '\n'.join(lines)
                explanation = f"Added initialization for undefined variable: {var_name}"
                confidence = 0.6
        
        return modified_code, explanation, confidence

class SelfModifyingCodeEngine:
    """
    Engine that coordinates autonomous code modification including:
    1. Analyzing code issues
    2. Generating improvements
    3. Safely applying changes
    4. Tracking modifications
    """
    
    def __init__(self):
        self.code_modifier = CodeModifier()
        self.ai_generator = AICodeGenerator()
        self.modification_queue = []
        self.modification_lock = threading.Lock()
        self.auto_mode_active = False
        self.auto_thread = None
        
        # Status tracking
        self.pending_issues = []
        self.load_pending_issues()
    
    def load_pending_issues(self):
        """Load pending issues from file"""
        issues_file = "data/learning/improvement_suggestions.json"
        
        if os.path.exists(issues_file):
            try:
                with open(issues_file, 'r') as file:
                    self.pending_issues = json.load(file)
                    logger.info(f"Loaded {len(self.pending_issues)} pending issues")
            except json.JSONDecodeError:
                logger.warning("Failed to load pending issues, starting fresh")
    
    def queue_modification(self, file_path: str, issue_description: str, modification_type: str = 'bugfix') -> bool:
        """
        Queue a file for improvement
        
        Args:
            file_path: Path to the file to modify
            issue_description: Description of the issue to fix
            modification_type: Type of modification ('bugfix', 'optimization', 'feature')
            
        Returns:
            bool: True if the modification was queued, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        with self.modification_lock:
            # Check if this file is already in the queue
            for queued in self.modification_queue:
                if queued["file_path"] == file_path:
                    logger.warning(f"File {file_path} is already in the modification queue")
                    return False
            
            # Queue the modification
            self.modification_queue.append({
                "file_path": file_path,
                "issue_description": issue_description,
                "modification_type": modification_type,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Queued {file_path} for {modification_type}")
            return True
    
    def process_queue(self, auto_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Process the modification queue
        
        Args:
            auto_mode: If True, automatically apply changes that pass safety checks
            
        Returns:
            list: Results of the processing
        """
        results = []
        
        with self.modification_lock:
            # Make a copy of the queue
            queue_copy = self.modification_queue.copy()
            self.modification_queue = []
        
        for item in queue_copy:
            file_path = item["file_path"]
            issue_description = item["issue_description"]
            modification_type = item["modification_type"]
            
            try:
                # Read the original code
                with open(file_path, 'r') as file:
                    original_code = file.read()
                
                # Generate improvement
                modified_code, explanation, confidence = self.ai_generator.generate_code_improvement(
                    file_path, original_code, issue_description
                )
                
                # Create the modification object
                modification = CodeModification(
                    file_path=file_path,
                    original_code=original_code,
                    modified_code=modified_code,
                    description=f"{issue_description}\n\n{explanation}",
                    modification_type=modification_type,
                    confidence=confidence
                )
                
                # Apply the modification if in auto mode
                if auto_mode and confidence >= self.code_modifier.min_confidence:
                    success = self.code_modifier.apply_modification(modification)
                    status = "applied" if success else "failed"
                else:
                    status = "generated"
                
                # Add to results
                results.append({
                    "file_path": file_path,
                    "status": status,
                    "confidence": confidence,
                    "description": issue_description,
                    "explanation": explanation,
                    "diff": modification.get_diff()
                })
                
                logger.info(f"Processed {file_path} with status {status}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e),
                    "description": issue_description
                })
        
        return results
    
    def start_auto_mode(self):
        """Start automated improvement mode"""
        if not self.auto_mode_active:
            self.auto_mode_active = True
            self.auto_thread = threading.Thread(target=self._auto_mode_loop)
            self.auto_thread.daemon = True
            self.auto_thread.start()
            logger.info("Started automatic code improvement mode")
            return True
        return False
    
    def stop_auto_mode(self):
        """Stop automated improvement mode"""
        if self.auto_mode_active:
            self.auto_mode_active = False
            if self.auto_thread:
                self.auto_thread.join(timeout=5.0)
            logger.info("Stopped automatic code improvement mode")
            return True
        return False
    
    def _auto_mode_loop(self):
        """Main loop for automatic improvement"""
        while self.auto_mode_active:
            try:
                # Check for pending issues
                if self.pending_issues:
                    for issue in self.pending_issues[:5]:  # Process up to 5 issues at a time
                        if "module" in issue and os.path.exists(issue["module"]):
                            self.queue_modification(
                                file_path=issue["module"],
                                issue_description=issue.get("description", "Unknown issue"),
                                modification_type="bugfix" if issue.get("severity") == "high" else "optimization"
                            )
                    
                    # Remove processed issues
                    self.pending_issues = self.pending_issues[5:]
                
                # Process the queue
                if self.modification_queue:
                    self.process_queue(auto_mode=True)
                
                # Sleep to prevent excessive processing
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto mode loop: {str(e)}")
                time.sleep(600)  # Longer sleep on error

# Initialize the global self-modifying code engine
self_modifying_code_engine = SelfModifyingCodeEngine()

# Utility function to get the engine instance
def get_self_modifying_code_engine():
    return self_modifying_code_engine

