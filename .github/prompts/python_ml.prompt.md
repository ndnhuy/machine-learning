# GitHub Copilot Python Machine Learning Prompt Guide

## Python Development Approach

When assisting me with Python code, especially for machine learning tasks, please:

1. **Structured Analysis**
   - Understand the ML problem type (classification, regression, clustering, etc.)
   - Identify data characteristics and requirements
   - Consider appropriate ML algorithms and approaches

2. **Code Organization**
   - Follow a modular architecture with clear separation of concerns
   - Create reusable components with well-defined interfaces
   - Use appropriate design patterns (e.g., Strategy, Factory, etc.)

3. **Python Best Practices**
   - Follow PEP 8 style guidelines
   - Use type hints (e.g., `def predict(X: np.ndarray) -> np.ndarray:`)
   - Implement proper error handling and validation
   - Include comprehensive docstrings in NumPy/Google format

4. **Machine Learning Standards**
   - Follow scikit-learn API conventions when appropriate
   - Implement standard ML interfaces (fit, predict, transform)
   - Include appropriate evaluation metrics and visualizations
   - Handle data preprocessing and feature engineering properly

5. **Performance Considerations**
   - Use vectorized operations rather than loops where possible
   - Consider memory constraints with large datasets
   - Implement appropriate optimization techniques
   - Use profiling when needed to identify bottlenecks

## Implementation Process

1. **Problem Definition**
   - Clearly state the problem and expected outcome
   - Define inputs, outputs, and constraints

2. **Component Design**
   - Break down the solution into clear components
   - Define interfaces between components
   - Consider extensibility and maintainability

3. **Code Implementation**
   - Start with minimal viable functionality
   - Add proper documentation and type hints
   - Include appropriate tests
   - Ensure error handling and edge cases are covered

4. **Refinement**
   - Optimize for performance where necessary
   - Improve readability and maintainability
   - Enhance documentation
   - Suggest potential improvements

## Python Code Style

- Use meaningful variable names (e.g., `learning_rate` not `lr`)
- Add docstrings to all functions, classes, and modules
- Include examples in docstrings for complex functions
- Use type annotations to improve code clarity
- Follow a consistent naming convention (snake_case for functions/variables, PascalCase for classes)
- Organize imports logically (standard library, third-party, local)
- Use appropriate whitespace and line breaks for readability

## Machine Learning Specific Guidelines

- Ensure reproducibility by setting random seeds
- Include data validation and preprocessing steps
- Implement proper train/validation/test splits
- Document hyperparameter choices and their impact
- Include evaluation metrics appropriate to the problem
- Visualize results when relevant
- Consider model interpretability
- Explain the tradeoffs of different algorithms

## Example Structure

```python
import numpy as np
from typing import Optional, Tuple, Union

class ModelVisualizer:
    """
    Interface for visualizing model predictions.
    
    This class provides methods to create visualizations of machine learning
    model predictions compared to actual data.
    
    Attributes:
        output_path: Path where visualizations will be saved
    """
    
    def visualize(self, 
                  x: np.ndarray, 
                  y: np.ndarray, 
                  y_pred: np.ndarray) -> None:
        """
        Visualize the actual data points and model predictions.
        
        Args:
            x: Feature values
            y: Actual target values
            y_pred: Predicted target values
            
        Returns:
            None
        """
        pass
```

---

Please follow these guidelines when helping me with Python machine learning tasks to ensure high-quality, maintainable, and educational responses.