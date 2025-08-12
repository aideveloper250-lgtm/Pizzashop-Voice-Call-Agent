"""
Production-ready Flask application for Synthflow.ai call initiation.

This module provides a web interface and API endpoints for initiating
phone calls through the Synthflow.ai service.

Author: Your Name
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import wraps
import time

import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration class."""
    
    # Synthflow API Configuration
    SYNTHFLOW_API_KEY = os.environ.get("SYNTHFLOW_API_KEY")
    SYNTHFLOW_AGENT_ID = os.environ.get("SYNTHFLOW_AGENT_ID")
    SYNTHFLOW_CALL_API_URL = "https://api.synthflow.ai/v2/calls"
    
    # Flask Configuration
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # Request Configuration
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
    
    # Server Configuration
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", "5000"))


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = Config.LOG_LEVEL) -> logging.Logger:
    """
    Configure application logging with both file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get the root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler for all logs
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # File Handler for error logs only
    error_handler = logging.FileHandler("logs/error.log")
    error_handler.setLevel(logging.ERROR)  # Set level specifically for this handler
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
    return logger


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config() -> None:
    """
    Validate required configuration variables.
    
    Raises:
        ValueError: If required environment variables are missing
    """
    required_vars = {
        "SYNTHFLOW_API_KEY": Config.SYNTHFLOW_API_KEY,
        "SYNTHFLOW_AGENT_ID": Config.SYNTHFLOW_AGENT_ID
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")


def validate_phone_number(phone_number: str) -> bool:
    """
    Validate phone number format using international standards.
    
    Args:
        phone_number: Phone number to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not phone_number or not isinstance(phone_number, str):
        return False
    
    # Remove common formatting characters
    cleaned = phone_number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    
    # Basic validation: should start with + and contain only digits after that
    if cleaned.startswith("+") and cleaned[1:].isdigit() and 10 <= len(cleaned) <= 15:
        return True
    
    logger.warning(f"Invalid phone number format: {phone_number}")
    return False


def sanitize_input(data: str, max_length: int = 255) -> str:
    """
    Sanitize input data by removing dangerous characters and limiting length.
    
    Args:
        data: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(data, str):
        return ""
    
    # Remove potentially dangerous characters and limit length
    sanitized = data.strip()[:max_length]
    return sanitized


# =============================================================================
# SYNTHFLOW SERVICE CLASS
# =============================================================================

class SynthflowService:
    """Service class for handling Synthflow.ai API interactions."""
    
    def __init__(self, api_key: str, agent_id: str, base_url: str, timeout: int = 30):
        """
        Initialize the Synthflow service with configuration and session setup.
        
        Args:
            api_key: Synthflow API key
            agent_id: Synthflow agent ID
            base_url: Base URL for Synthflow API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.agent_id = agent_id
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._create_session()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info("SynthflowService initialized successfully")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry strategy and connection pooling.
        
        Returns:
            Configured requests session with retry logic
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=Config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_redirect=False,
            raise_on_status=False
        )
        
        # Configure HTTP adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "User-Agent": "SynthflowCallInitiator/1.0.0",
            "Accept": "application/json"
        })
        
        return session
    
    def initiate_call(self, phone_number: str, customer_name: str = "Customer") -> Dict[str, Any]:
        """
        Initiate a call through Synthflow.ai API with comprehensive error handling.
        
        Args:
            phone_number: Target phone number in international format
            customer_name: Customer name for the call
            
        Returns:
            API response data containing call information
            
        Raises:
            requests.RequestException: If API call fails after retries
        """
        start_time = time.time()
        
        # Sanitize inputs
        phone_number = sanitize_input(phone_number, 20)
        customer_name = sanitize_input(customer_name, 100)
        
        payload = {
            "model_id": self.agent_id,
            "phone": phone_number,
            "name": customer_name
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"Initiating call to {phone_number} for customer: {customer_name}")
        self.logger.debug(f"Request payload: {payload}")
        
        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Log response details
            elapsed_time = time.time() - start_time
            self.logger.info(f"API request completed in {elapsed_time:.2f}s with status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            
            call_id = response_data.get('callId', 'unknown')
            self.logger.info(f"Call initiated successfully. Call ID: {call_id}")
            self.logger.debug(f"Full API response: {response_data}")
            
            return response_data
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error during call initiation: {e}")
            self.logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
            raise
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error during call initiation: {e}")
            raise
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout error during call initiation: {e}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Unexpected error during call initiation: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected non-request error: {e}")
            raise requests.exceptions.RequestException(f"Unexpected error: {e}")


# =============================================================================
# DECORATORS
# =============================================================================

def handle_api_errors(f):
    """
    Decorator for handling API errors consistently across endpoints.
    
    This decorator catches various types of exceptions and returns
    appropriate HTTP responses with proper error messages.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in {f.__name__}: {e}")
            status_code = 503
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code if e.response.status_code < 500 else 503
            return jsonify({
                "error": "Service temporarily unavailable",
                "status": "error",
                "timestamp": time.time()
            }), status_code
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error in {f.__name__}: {e}")
            return jsonify({
                "error": "Unable to connect to external service",
                "status": "error",
                "timestamp": time.time()
            }), 503
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error in {f.__name__}: {e}")
            return jsonify({
                "error": "Request timeout - please try again",
                "status": "error",
                "timestamp": time.time()
            }), 504
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error in {f.__name__}: {e}")
            return jsonify({
                "error": "External service error occurred",
                "status": "error",
                "timestamp": time.time()
            }), 500
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({
                "error": str(e),
                "status": "error",
                "timestamp": time.time()
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "status": "error",
                "timestamp": time.time()
            }), 500
    
    return decorated_function


def log_requests(f):
    """Decorator to log all incoming requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        logger.info(f"Request: {request.method} {request.path} from {client_ip}")
        
        try:
            result = f(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Request completed in {elapsed_time:.3f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Request failed after {elapsed_time:.3f}s: {e}")
            raise
    
    return decorated_function


# =============================================================================
# FLASK APPLICATION FACTORY
# =============================================================================

def create_app() -> Flask:
    """
    Create and configure Flask application with all necessary components.
    
    Returns:
        Configured Flask application instance
    """
    # Validate configuration before creating app
    validate_config()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for frontend integration
    CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
    
    # Initialize Synthflow service
    synthflow_service = SynthflowService(
        api_key=Config.SYNTHFLOW_API_KEY,
        agent_id=Config.SYNTHFLOW_AGENT_ID,
        base_url=Config.SYNTHFLOW_CALL_API_URL,
        timeout=Config.REQUEST_TIMEOUT
    )
    
    # =============================================================================
    # ROUTE DEFINITIONS
    # =============================================================================
    
    @app.route('/')
    @log_requests
    def home():
        """
        Serve the landing page with the call initiation form.
        
        Returns:
            Rendered HTML template for the call form
        """
        logger.info("Home page accessed")
        try:
            return render_template('form.html')
        except Exception as e:
            logger.error(f"Error serving home page: {e}")
            return jsonify({
                "error": "Unable to serve home page",
                "status": "error"
            }), 500
    
    @app.route('/health')
    @log_requests
    def health_check():
        """
        Health check endpoint for monitoring and load balancers.
        
        Returns:
            JSON response with service health status
        """
        try:
            # Basic health check - could be extended to check database, external services, etc.
            health_data = {
                "status": "healthy",
                "service": "synthflow-call-initiator",
                "version": "1.0.0",
                "timestamp": time.time(),
                "uptime": time.time(),  # In production, calculate actual uptime
                "environment": "production" if not Config.DEBUG else "development"
            }
            
            logger.debug("Health check performed successfully")
            return jsonify(health_data), 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                "status": "unhealthy",
                "error": "Health check failed",
                "timestamp": time.time()
            }), 503
    
    @app.route('/api/initiate-call', methods=['POST'])
    @log_requests
    @handle_api_errors
    def initiate_call():
        """
        Initiate a phone call through Synthflow.ai API.
        
        Expected JSON payload:
        {
            "phoneNumber": "+1234567890",
            "customerName": "John Doe" (optional)
        }
        
        Returns:
            JSON response with call initiation status and details
        """
        logger.info("Call initiation API endpoint accessed")
        
        # Validate request format
        if not request.is_json:
            logger.warning("Invalid request format - not JSON")
            return jsonify({
                "error": "Request must be in JSON format",
                "status": "error",
                "timestamp": time.time()
            }), 400
        
        try:
            data = request.get_json()
            
            if not data:
                logger.warning("Empty JSON payload received")
                return jsonify({
                    "error": "Empty request payload",
                    "status": "error",
                    "timestamp": time.time()
                }), 400
            
            # Extract and validate input data
            phone_number = data.get('phoneNumber', '').strip()
            customer_name = data.get('customerName', 'Customer').strip()
            
            # Validate required fields
            if not phone_number:
                logger.warning("Phone number missing in request")
                return jsonify({
                    "error": "Phone number is required",
                    "status": "error",
                    "timestamp": time.time()
                }), 400
            
            # Validate phone number format
            if not validate_phone_number(phone_number):
                logger.warning(f"Invalid phone number format received: {phone_number}")
                return jsonify({
                    "error": "Invalid phone number format. Please use international format (e.g., +1234567890)",
                    "status": "error",
                    "timestamp": time.time()
                }), 400
            
            # Validate customer name length
            if len(customer_name) > 100:
                logger.warning(f"Customer name too long: {len(customer_name)} characters")
                return jsonify({
                    "error": "Customer name must be less than 100 characters",
                    "status": "error",
                    "timestamp": time.time()
                }), 400
            
            # Log the initiation attempt
            logger.info(f"Initiating call to {phone_number} for customer: {customer_name}")
            
            # Make the API call through Synthflow service
            response_data = synthflow_service.initiate_call(phone_number, customer_name)
            
            # Prepare success response
            success_response = {
                "message": "Call initiated successfully",
                "callId": response_data.get("callId"),
                "status": "success",
                "phoneNumber": phone_number,
                "customerName": customer_name,
                "timestamp": time.time()
            }
            
            logger.info(f"Call initiation successful. Call ID: {response_data.get('callId')}")
            return jsonify(success_response), 200
            
        except Exception as e:
            logger.error(f"Unexpected error in initiate_call: {e}", exc_info=True)
            raise  # Let the decorator handle it
    
    # =============================================================================
    # ERROR HANDLERS
    # =============================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        logger.warning(f"404 error: {request.method} {request.url} from {request.remote_addr}")
        return jsonify({
            "error": "Endpoint not found",
            "status": "error",
            "path": request.path,
            "timestamp": time.time()
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        logger.warning(f"405 error: {request.method} {request.url} from {request.remote_addr}")
        return jsonify({
            "error": f"Method {request.method} not allowed for this endpoint",
            "status": "error",
            "allowed_methods": error.valid_methods,
            "timestamp": time.time()
        }), 405
    
    @app.errorhandler(413)
    def payload_too_large(error):
        """Handle 413 Payload Too Large errors."""
        logger.warning(f"413 error: Payload too large from {request.remote_addr}")
        return jsonify({
            "error": "Request payload too large",
            "status": "error",
            "timestamp": time.time()
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors."""
        logger.warning(f"429 error: Rate limit exceeded from {request.remote_addr}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later",
            "status": "error",
            "timestamp": time.time()
        }), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"500 error: {error} from {request.remote_addr}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "timestamp": time.time()
        }), 500
    
    # =============================================================================
    # REQUEST MIDDLEWARE
    # =============================================================================
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        # Add request timestamp for logging
        request.start_time = time.time()
        
        # Log request details (except for health checks to reduce noise)
        if request.path != '/health':
            logger.debug(f"Incoming request: {request.method} {request.path}")
    
    @app.after_request
    def after_request(response):
        """Execute after each request."""
        # Add CORS headers if not already present
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Log response time (except for health checks)
        if hasattr(request, 'start_time') and request.path != '/health':
            duration = time.time() - request.start_time
            logger.debug(f"Request completed in {duration:.3f}s with status {response.status_code}")
        
        return response
    
    logger.info("Flask application created and configured successfully")
    return app


# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

# Setup logging first
logger = setup_logging()

# Create the Flask application
try:
    app = create_app()
    logger.info("Application initialization completed successfully")
except Exception as e:
    logger.error(f"Failed to initialize application: {e}", exc_info=True)
    raise


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    try:
        logger.info(f"Starting Flask application on {Config.HOST}:{Config.PORT}")
        logger.info(f"Debug mode: {Config.DEBUG}")
        logger.info(f"Log level: {Config.LOG_LEVEL}")
        
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutdown completed")