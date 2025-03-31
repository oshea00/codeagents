# main.py
"""
Weather Forecast API with FastAPI

This application retrieves weather forecasts from the National Weather Service (NWS) API.
It accepts location input in two formats: city/state or latitude/longitude coordinates,
and handles the two-step process required by the NWS API to fetch forecast data.

Implementation Approach:
1. Use FastAPI for creating API endpoints with automatic documentation
2. Implement geocoding service to convert city/state to coordinates using GeoPy
3. Create an HTTP client with redirect handling using httpx
4. Develop service layer to interact with the NWS API
5. Define Pydantic models for request/response validation
6. Implement proper error handling and input validation
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union, Any, Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from pydantic import BaseModel, Field, field_validator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----- Pydantic Models -----

class LocationByCity(BaseModel):
    """Model for location input by city and state."""
    city: str
    state: str

    @field_validator('city')
    @classmethod
    def city_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('City cannot be empty')
        return v.strip()

    @field_validator('state')
    @classmethod
    def state_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('State cannot be empty')
        return v.strip()


class LocationByCoords(BaseModel):
    """Model for location input by latitude and longitude."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    lon: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")


class ForecastPeriod(BaseModel):
    """Model for a single forecast period."""
    number: int
    name: str
    startTime: str
    endTime: str
    isDaytime: bool
    temperature: int
    temperatureUnit: str
    temperatureTrend: Optional[str] = None
    windSpeed: str
    windDirection: str
    icon: str
    shortForecast: str
    detailedForecast: str


class ForecastResponse(BaseModel):
    """Model for the complete forecast response."""
    location: str
    periods: List[ForecastPeriod]
    updated: str
    generated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str


# ----- Service Layer -----

class GeocodingService:
    """Service for converting city/state to latitude/longitude coordinates."""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="weather-forecast-api")
        self.cache = {}  # Simple in-memory cache
    
    async def geocode(self, city: str, state: str) -> Tuple[float, float]:
        """
        Convert city and state to latitude and longitude coordinates.
        
        Args:
            city: City name
            state: State name or abbreviation
            
        Returns:
            Tuple containing latitude and longitude
            
        Raises:
            HTTPException: If location cannot be found or geocoding service fails
        """
        cache_key = f"{city.lower()},{state.lower()}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        try:
            # Run geocoding in a separate thread to avoid blocking
            location = await asyncio.to_thread(
                self.geolocator.geocode, f"{city}, {state}, USA", exactly_one=True
            )
            
            if not location:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Location not found for {city}, {state}"
                )
            
            coordinates = (location.latitude, location.longitude)
            
            # Cache the result
            self.cache[cache_key] = coordinates
            logger.info(f"Geocoded {city}, {state} to {coordinates}")
            
            return coordinates
            
        except GeocoderTimedOut:
            logger.error(f"Geocoding timeout for {city}, {state}")
            raise HTTPException(
                status_code=503, 
                detail="Geocoding service timed out, please try again later"
            )
        except GeocoderServiceError as e:
            logger.error(f"Geocoding service error: {str(e)}")
            raise HTTPException(
                status_code=503, 
                detail="Geocoding service error, please try again later"
            )
        except Exception as e:
            logger.error(f"Unexpected error during geocoding: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail="An unexpected error occurred during geocoding"
            )


class WeatherService:
    """Service for interacting with the National Weather Service API."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "WeatherForecastAPI/1.0 (github.com/example/weather-api; contact@example.com)",
                "Accept": "application/geo+json"
            }
        )
        self.points_cache = {}  # Cache for grid points
        self.forecast_cache = {}  # Cache for forecasts
        self.cache_ttl = 3600  # Cache time-to-live in seconds (1 hour)
    
    async def get_grid_point(self, lat: float, lon: float) -> Dict:
        """
        Get grid point information from NWS API for the given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary containing grid point data
            
        Raises:
            HTTPException: If API request fails
        """
        cache_key = f"{lat:.4f},{lon:.4f}"
        
        # Check cache first
        if cache_key in self.points_cache:
            cache_entry = self.points_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                logger.info(f"Grid point cache hit for {cache_key}")
                return cache_entry["data"]
        
        try:
            url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            logger.info(f"Fetching grid point data from {url}")
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self.points_cache[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
            
            return data
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(f"HTTP error {status_code} when fetching grid point: {str(e)}")
            
            if status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"No weather data available for coordinates: {lat}, {lon}"
                )
            elif status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests to weather service, please try again later"
                )
            else:
                raise HTTPException(
                    status_code=status_code,
                    detail=f"Weather service error: {e.response.text}"
                )
                
        except httpx.RequestError as e:
            logger.error(f"Request error when fetching grid point: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Unable to connect to weather service, please try again later"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error when fetching grid point: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while fetching weather data"
            )
    
    async def get_forecast(self, grid_data: Dict) -> Dict:
        """
        Get forecast data from NWS API using grid point information.
        
        Args:
            grid_data: Grid point data from get_grid_point
            
        Returns:
            Dictionary containing forecast data
            
        Raises:
            HTTPException: If API request fails
        """
        try:
            forecast_url = grid_data["properties"]["forecast"]
            
            # Check cache first
            if forecast_url in self.forecast_cache:
                cache_entry = self.forecast_cache[forecast_url]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    logger.info(f"Forecast cache hit for {forecast_url}")
                    return cache_entry["data"]
            
            logger.info(f"Fetching forecast data from {forecast_url}")
            response = await self.client.get(forecast_url)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self.forecast_cache[forecast_url] = {
                "data": data,
                "timestamp": time.time()
            }
            
            return data
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(f"HTTP error {status_code} when fetching forecast: {str(e)}")
            
            if status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Forecast data not found"
                )
            elif status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests to weather service, please try again later"
                )
            else:
                raise HTTPException(
                    status_code=status_code,
                    detail=f"Weather service error: {e.response.text}"
                )
                
        except httpx.RequestError as e:
            logger.error(f"Request error when fetching forecast: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Unable to connect to weather service, please try again later"
            )
            
        except KeyError as e:
            logger.error(f"Missing key in grid data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Invalid grid data format from weather service"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error when fetching forecast: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while fetching forecast data"
            )
    
    async def format_forecast_response(self, grid_data: Dict, forecast_data: Dict) -> ForecastResponse:
        """
        Format the raw forecast data into a structured response.
        
        Args:
            grid_data: Grid point data from get_grid_point
            forecast_data: Forecast data from get_forecast
            
        Returns:
            Formatted ForecastResponse object
        """
        try:
            # Extract location information
            relative_location = grid_data["properties"]["relativeLocation"]["properties"]
            location = f"{relative_location['city']}, {relative_location['state']}"
            
            # Extract forecast periods
            periods = forecast_data["properties"]["periods"]
            
            # Extract update time
            updated = forecast_data["properties"]["updateTime"]
            
            return ForecastResponse(
                location=location,
                periods=periods,
                updated=updated
            )
            
        except KeyError as e:
            logger.error(f"Missing key in API response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid data format in weather service response: missing {str(e)}"
            )
    
    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()


# ----- Services Initialization -----

# We'll create the services at module level for normal operation
# but allow them to be overridden for testing through dependency injection
weather_service = WeatherService()
geocoding_service = GeocodingService()

# ----- Application Lifecycle -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    This replaces the deprecated on_event handlers.
    """
    # Services are already initialized at module level
    logger.info("Weather Forecast API starting up")
    
    yield
    
    # Clean up resources on shutdown
    logger.info("Weather Forecast API shutting down")
    await weather_service.close()

# Initialize FastAPI application with lifespan
app = FastAPI(
    title="Weather Forecast API",
    description="API for retrieving weather forecasts from the National Weather Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Dependency Injection -----

def get_geocoding_service() -> GeocodingService:
    """
    Dependency for geocoding service.
    In normal operation, returns the global instance.
    For testing, this can be overridden with a mock.
    """
    return geocoding_service


def get_weather_service() -> WeatherService:
    """
    Dependency for weather service.
    In normal operation, returns the global instance.
    For testing, this can be overridden with a mock.
    """
    return weather_service


# ----- API Routes -----

@app.get(
    "/forecast/coordinates/",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Get forecast by coordinates",
    description="Retrieve weather forecast for the specified latitude and longitude coordinates"
)
async def forecast_by_coordinates(
    lat: Annotated[float, Query(..., ge=-90, le=90, description="Latitude between -90 and 90")],
    lon: Annotated[float, Query(..., ge=-180, le=180, description="Longitude between -180 and 180")],
    weather_service: WeatherService = Depends(get_weather_service)
):
    """
    Get weather forecast for the specified latitude and longitude coordinates.
    
    Args:
        lat: Latitude coordinate (between -90 and 90)
        lon: Longitude coordinate (between -180 and 180)
        weather_service: Injected WeatherService instance
        
    Returns:
        ForecastResponse containing the weather forecast
    """
    logger.info(f"Processing forecast request for coordinates: {lat}, {lon}")
    
    try:
        grid_data = await weather_service.get_grid_point(lat, lon)
        forecast_data = await weather_service.get_forecast(grid_data)
        
        return await weather_service.format_forecast_response(grid_data, forecast_data)
    except HTTPException:
        # Re-raise HTTP exceptions directly to preserve status code and message
        raise
    except Exception as e:
        # Log unexpected errors and convert to HTTPException
        logger.error(f"Unexpected error in forecast_by_coordinates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get(
    "/forecast/location/",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Get forecast by city and state",
    description="Retrieve weather forecast for the specified city and state"
)
async def forecast_by_location(
    city: Annotated[str, Query(..., min_length=1, description="City name")],
    state: Annotated[str, Query(..., min_length=1, description="State name or abbreviation")],
    geocoding_service: GeocodingService = Depends(get_geocoding_service),
    weather_service: WeatherService = Depends(get_weather_service)
):
    """
    Get weather forecast for the specified city and state.
    
    Args:
        city: City name
        state: State name or abbreviation
        geocoding_service: Injected GeocodingService instance
        weather_service: Injected WeatherService instance
        
    Returns:
        ForecastResponse containing the weather forecast
    """
    logger.info(f"Processing forecast request for location: {city}, {state}")
    
    try:
        # Convert city and state to coordinates
        lat, lon = await geocoding_service.geocode(city, state)
        
        # Get forecast using coordinates
        grid_data = await weather_service.get_grid_point(lat, lon)
        forecast_data = await weather_service.get_forecast(grid_data)
        
        return await weather_service.format_forecast_response(grid_data, forecast_data)
    except HTTPException:
        # Re-raise HTTP exceptions directly to preserve status code and message
        raise
    except Exception as e:
        # Log unexpected errors and convert to HTTPException
        logger.error(f"Unexpected error in forecast_by_location: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post(
    "/forecast/coordinates/",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Get forecast by coordinates (POST)",
    description="Retrieve weather forecast for the specified latitude and longitude coordinates using POST"
)
async def forecast_by_coordinates_post(
    location: LocationByCoords,
    weather_service: WeatherService = Depends(get_weather_service)
):
    """
    Get weather forecast for the specified latitude and longitude coordinates using POST.
    
    Args:
        location: LocationByCoords model containing lat and lon
        weather_service: Injected WeatherService instance
        
    Returns:
        ForecastResponse containing the weather forecast
    """
    logger.info(f"Processing POST forecast request for coordinates: {location.lat}, {location.lon}")
    
    try:
        grid_data = await weather_service.get_grid_point(location.lat, location.lon)
        forecast_data = await weather_service.get_forecast(grid_data)
        
        return await weather_service.format_forecast_response(grid_data, forecast_data)
    except HTTPException:
        # Re-raise HTTP exceptions directly to preserve status code and message
        raise
    except Exception as e:
        # Log unexpected errors and convert to HTTPException
        logger.error(f"Unexpected error in forecast_by_coordinates_post: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post(
    "/forecast/location/",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Get forecast by city and state (POST)",
    description="Retrieve weather forecast for the specified city and state using POST"
)
async def forecast_by_location_post(
    location: LocationByCity,
    geocoding_service: GeocodingService = Depends(get_geocoding_service),
    weather_service: WeatherService = Depends(get_weather_service)
):
    """
    Get weather forecast for the specified city and state using POST.
    
    Args:
        location: LocationByCity model containing city and state
        geocoding_service: Injected GeocodingService instance
        weather_service: Injected WeatherService instance
        
    Returns:
        ForecastResponse containing the weather forecast
    """
    logger.info(f"Processing POST forecast request for location: {location.city}, {location.state}")
    
    try:
        # Convert city and state to coordinates
        lat, lon = await geocoding_service.geocode(location.city, location.state)
        
        # Get forecast using coordinates
        grid_data = await weather_service.get_grid_point(lat, lon)
        forecast_data = await weather_service.get_forecast(grid_data)
        
        return await weather_service.format_forecast_response(grid_data, forecast_data)
    except HTTPException:
        # Re-raise HTTP exceptions directly to preserve status code and message
        raise
    except Exception as e:
        # Log unexpected errors and convert to HTTPException
        logger.error(f"Unexpected error in forecast_by_location_post: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


# ----- Test Configuration Helper -----

def setup_test_config():
    """
    Configure the application for testing.
    This function is called by pytest fixtures to set up the test environment.
    """
    # This is a hook for pytest to configure the application for testing
    # It can be used to set any test-specific configurations
    import warnings
    warnings.filterwarnings(
        "ignore", 
        message="The configuration option \"asyncio_default_fixture_loop_scope\" is unset"
    )


# ----- Main Entry Point -----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)