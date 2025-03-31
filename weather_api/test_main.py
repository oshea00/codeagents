# test_main.py
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from main import (
    app, 
    GeocodingService, 
    WeatherService, 
    LocationByCity, 
    LocationByCoords, 
    ForecastResponse, 
    ForecastPeriod,
    setup_test_config
)


@pytest.fixture
def client():
    setup_test_config()
    return TestClient(app)


@pytest.fixture
def mock_geocoding_service():
    service = GeocodingService()
    service.geocode = AsyncMock()
    service.geocode.return_value = (37.7749, -122.4194)  # San Francisco coordinates
    return service


@pytest.fixture
def mock_weather_service():
    service = WeatherService()
    service.get_grid_point = AsyncMock()
    service.get_forecast = AsyncMock()
    service.format_forecast_response = AsyncMock()
    service.close = AsyncMock()
    
    # Set up sample responses
    grid_data = {
        "properties": {
            "forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast",
            "relativeLocation": {
                "properties": {
                    "city": "San Francisco",
                    "state": "CA"
                }
            }
        }
    }
    
    forecast_data = {
        "properties": {
            "periods": [
                {
                    "number": 1,
                    "name": "Tonight",
                    "startTime": "2023-01-01T18:00:00-08:00",
                    "endTime": "2023-01-02T06:00:00-08:00",
                    "isDaytime": False,
                    "temperature": 55,
                    "temperatureUnit": "F",
                    "temperatureTrend": None,
                    "windSpeed": "5 mph",
                    "windDirection": "SW",
                    "icon": "https://api.weather.gov/icons/land/night/few?size=medium",
                    "shortForecast": "Mostly Clear",
                    "detailedForecast": "Mostly clear, with a low around 55."
                }
            ],
            "updateTime": "2023-01-01T18:00:00-08:00"
        }
    }
    
    forecast_response = ForecastResponse(
        location="San Francisco, CA",
        periods=[
            ForecastPeriod(
                number=1,
                name="Tonight",
                startTime="2023-01-01T18:00:00-08:00",
                endTime="2023-01-02T06:00:00-08:00",
                isDaytime=False,
                temperature=55,
                temperatureUnit="F",
                temperatureTrend=None,
                windSpeed="5 mph",
                windDirection="SW",
                icon="https://api.weather.gov/icons/land/night/few?size=medium",
                shortForecast="Mostly Clear",
                detailedForecast="Mostly clear, with a low around 55."
            )
        ],
        updated="2023-01-01T18:00:00-08:00",
        generated_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    service.get_grid_point.return_value = grid_data
    service.get_forecast.return_value = forecast_data
    service.format_forecast_response.return_value = forecast_response
    
    return service


class TestGeocodingService:
    @pytest.fixture
    def geocoding_service(self):
        return GeocodingService()
    
    @pytest.mark.asyncio
    async def test_geocode_success(self, geocoding_service):
        with patch('asyncio.to_thread') as mock_to_thread:
            # Mock the geopy location result
            mock_location = MagicMock()
            mock_location.latitude = 37.7749
            mock_location.longitude = -122.4194
            mock_to_thread.return_value = mock_location
            
            result = await geocoding_service.geocode("San Francisco", "CA")
            
            assert result == (37.7749, -122.4194)
            assert geocoding_service.cache.get("san francisco,ca") == (37.7749, -122.4194)
    
    @pytest.mark.asyncio
    async def test_geocode_cache_hit(self, geocoding_service):
        # Manually add to cache
        geocoding_service.cache["san francisco,ca"] = (37.7749, -122.4194)
        
        result = await geocoding_service.geocode("San Francisco", "CA")
        
        assert result == (37.7749, -122.4194)
    
    @pytest.mark.asyncio
    async def test_geocode_location_not_found(self, geocoding_service):
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            with pytest.raises(HTTPException) as excinfo:
                await geocoding_service.geocode("NonExistent", "XX")
            
            assert excinfo.value.status_code == 404
            assert "Location not found" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_geocode_timeout(self, geocoding_service):
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = GeocoderTimedOut("Timeout")
            
            with pytest.raises(HTTPException) as excinfo:
                await geocoding_service.geocode("San Francisco", "CA")
            
            assert excinfo.value.status_code == 503
            assert "timed out" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_geocode_service_error(self, geocoding_service):
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = GeocoderServiceError("Service Error")
            
            with pytest.raises(HTTPException) as excinfo:
                await geocoding_service.geocode("San Francisco", "CA")
            
            assert excinfo.value.status_code == 503
            assert "service error" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_geocode_unexpected_error(self, geocoding_service):
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("Unexpected error")
            
            with pytest.raises(HTTPException) as excinfo:
                await geocoding_service.geocode("San Francisco", "CA")
            
            assert excinfo.value.status_code == 500
            assert "unexpected error" in excinfo.value.detail


class TestWeatherService:
    @pytest.fixture
    def weather_service(self):
        service = WeatherService()
        return service
    
    @pytest.mark.asyncio
    async def test_get_grid_point_success(self, weather_service):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
        
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await weather_service.get_grid_point(37.7749, -122.4194)
            
            mock_get.assert_called_once_with('https://api.weather.gov/points/37.7749,-122.4194')
            assert result == {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
            
            # Test cache
            assert len(weather_service.points_cache) == 1
            cache_key = "37.7749,-122.4194"
            assert cache_key in weather_service.points_cache
            assert weather_service.points_cache[cache_key]["data"] == result
    
    @pytest.mark.asyncio
    async def test_get_grid_point_cache_hit(self, weather_service):
        cache_key = "37.7749,-122.4194"
        cached_data = {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
        weather_service.points_cache[cache_key] = {
            "data": cached_data,
            "timestamp": time.time()
        }
        
        result = await weather_service.get_grid_point(37.7749, -122.4194)
        
        assert result == cached_data
    
    @pytest.mark.asyncio
    async def test_get_grid_point_http_status_error(self, weather_service):
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Error", 
                request=MagicMock(), 
                response=MagicMock(status_code=404, text="Not Found")
            )
            
            with pytest.raises(HTTPException) as excinfo:
                await weather_service.get_grid_point(37.7749, -122.4194)
            
            assert excinfo.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_grid_point_request_error(self, weather_service):
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection error", request=MagicMock())
            
            with pytest.raises(HTTPException) as excinfo:
                await weather_service.get_grid_point(37.7749, -122.4194)
            
            assert excinfo.value.status_code == 503
            assert "Unable to connect" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_get_forecast_success(self, weather_service):
        grid_data = {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
        forecast_data = {
            "properties": {
                "periods": [
                    {
                        "number": 1,
                        "name": "Tonight",
                        "startTime": "2023-01-01T18:00:00-08:00",
                        "endTime": "2023-01-02T06:00:00-08:00",
                        "isDaytime": False,
                        "temperature": 55,
                        "temperatureUnit": "F",
                        "temperatureTrend": None,
                        "windSpeed": "5 mph",
                        "windDirection": "SW",
                        "icon": "https://api.weather.gov/icons/land/night/few?size=medium",
                        "shortForecast": "Mostly Clear",
                        "detailedForecast": "Mostly clear, with a low around 55."
                    }
                ]
            }
        }
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = forecast_data
        
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await weather_service.get_forecast(grid_data)
            
            mock_get.assert_called_once_with('https://api.weather.gov/gridpoints/MTR/84,105/forecast')
            assert result == forecast_data
            
            # Test cache
            forecast_url = grid_data["properties"]["forecast"]
            assert forecast_url in weather_service.forecast_cache
            assert weather_service.forecast_cache[forecast_url]["data"] == forecast_data
    
    @pytest.mark.asyncio
    async def test_get_forecast_cache_hit(self, weather_service):
        forecast_url = "https://api.weather.gov/gridpoints/MTR/84,105/forecast"
        grid_data = {"properties": {"forecast": forecast_url}}
        forecast_data = {"properties": {"periods": []}}
        
        weather_service.forecast_cache[forecast_url] = {
            "data": forecast_data,
            "timestamp": time.time()
        }
        
        result = await weather_service.get_forecast(grid_data)
        
        assert result == forecast_data
    
    @pytest.mark.asyncio
    async def test_get_forecast_http_status_error(self, weather_service):
        grid_data = {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
        
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Error", 
                request=MagicMock(), 
                response=MagicMock(status_code=404, text="Not Found")
            )
            
            with pytest.raises(HTTPException) as excinfo:
                await weather_service.get_forecast(grid_data)
            
            assert excinfo.value.status_code == 404
            assert "not found" in excinfo.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_forecast_request_error(self, weather_service):
        grid_data = {"properties": {"forecast": "https://api.weather.gov/gridpoints/MTR/84,105/forecast"}}
        
        with patch.object(weather_service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection error", request=MagicMock())
            
            with pytest.raises(HTTPException) as excinfo:
                await weather_service.get_forecast(grid_data)
            
            assert excinfo.value.status_code == 503
            assert "Unable to connect" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_get_forecast_key_error(self, weather_service):
        grid_data = {"properties": {}}  # Missing forecast key
        
        with pytest.raises(HTTPException) as excinfo:
            await weather_service.get_forecast(grid_data)
        
        assert excinfo.value.status_code == 500
        assert "Invalid grid data format" in excinfo.value.detail
    
    @pytest.mark.asyncio
    async def test_format_forecast_response(self, weather_service):
        grid_data = {
            "properties": {
                "relativeLocation": {
                    "properties": {
                        "city": "San Francisco",
                        "state": "CA"
                    }
                }
            }
        }
        
        forecast_data = {
            "properties": {
                "periods": [
                    {
                        "number": 1,
                        "name": "Tonight",
                        "startTime": "2023-01-01T18:00:00-08:00",
                        "endTime": "2023-01-02T06:00:00-08:00",
                        "isDaytime": False,
                        "temperature": 55,
                        "temperatureUnit": "F",
                        "temperatureTrend": None,
                        "windSpeed": "5 mph",
                        "windDirection": "SW",
                        "icon": "https://api.weather.gov/icons/land/night/few?size=medium",
                        "shortForecast": "Mostly Clear",
                        "detailedForecast": "Mostly clear, with a low around 55."
                    }
                ],
                "updateTime": "2023-01-01T18:00:00-08:00"
            }
        }
        
        result = await weather_service.format_forecast_response(grid_data, forecast_data)
        
        assert isinstance(result, ForecastResponse)
        assert result.location == "San Francisco, CA"
        assert len(result.periods) == 1
        assert result.periods[0].number == 1
        assert result.periods[0].name == "Tonight"
        assert result.periods[0].temperature == 55
        assert result.updated == "2023-01-01T18:00:00-08:00"
    
    @pytest.mark.asyncio
    async def test_format_forecast_response_key_error(self, weather_service):
        grid_data = {"properties": {}}  # Missing relativeLocation
        forecast_data = {"properties": {"periods": [], "updateTime": "2023-01-01T18:00:00-08:00"}}
        
        with pytest.raises(HTTPException) as excinfo:
            await weather_service.format_forecast_response(grid_data, forecast_data)
        
        assert excinfo.value.status_code == 500
        assert "Invalid data format" in excinfo.value.detail


class TestModelValidation:
    def test_location_by_city_validation(self):
        # Valid data
        location = LocationByCity(city="San Francisco", state="CA")
        assert location.city == "San Francisco"
        assert location.state == "CA"
        
        # Test whitespace stripping
        location = LocationByCity(city=" San Francisco ", state=" CA ")
        assert location.city == "San Francisco"
        assert location.state == "CA"
        
        # Test validation errors
        with pytest.raises(ValueError):
            LocationByCity(city="", state="CA")
        
        with pytest.raises(ValueError):
            LocationByCity(city="San Francisco", state="")
    
    def test_location_by_coords_validation(self):
        # Valid data
        location = LocationByCoords(lat=37.7749, lon=-122.4194)
        assert location.lat == 37.7749
        assert location.lon == -122.4194
        
        # Test validation errors - out of range
        with pytest.raises(ValueError):
            LocationByCoords(lat=91.0, lon=-122.4194)
        
        with pytest.raises(ValueError):
            LocationByCoords(lat=37.7749, lon=181.0)


class TestAPIRoutes:
    @pytest.mark.asyncio
    async def test_forecast_by_coordinates(self, client, mock_weather_service):
        with patch('main.get_weather_service', return_value=mock_weather_service):
            response = client.get("/forecast/coordinates/?lat=37.7749&lon=-122.4194")
            
            assert response.status_code == 200
            data = response.json()
            assert data["location"] == "San Francisco, CA"
            assert len(data["periods"]) == 1
            assert data["periods"][0]["temperature"] == 55
    
    @pytest.mark.asyncio
    async def test_forecast_by_coordinates_invalid_params(self, client):
        response = client.get("/forecast/coordinates/?lat=91&lon=-122.4194")
        assert response.status_code == 422  # Validation error
        
        response = client.get("/forecast/coordinates/?lat=37.7749&lon=181")
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_forecast_by_location(self, client, mock_geocoding_service, mock_weather_service):
        with patch('main.get_geocoding_service', return_value=mock_geocoding_service), \
             patch('main.get_weather_service', return_value=mock_weather_service):
            response = client.get("/forecast/location/?city=San%20Francisco&state=CA")
            
            assert response.status_code == 200
            data = response.json()
            assert data["location"] == "San Francisco, CA"
            assert len(data["periods"]) == 1
            assert data["periods"][0]["temperature"] == 55
    
    @pytest.mark.asyncio
    async def test_forecast_by_location_invalid_params(self, client):
        response = client.get("/forecast/location/?city=&state=CA")
        assert response.status_code == 422  # Validation error
        
        response = client.get("/forecast/location/?city=San%20Francisco&state=")
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_forecast_by_coordinates_post(self, client, mock_weather_service):
        with patch('main.get_weather_service', return_value=mock_weather_service):
            response = client.post(
                "/forecast/coordinates/",
                json={"lat": 37.7749, "lon": -122.4194}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["location"] == "San Francisco, CA"
            assert len(data["periods"]) == 1
            assert data["periods"][0]["temperature"] == 55
    
    @pytest.mark.asyncio
    async def test_forecast_by_coordinates_post_invalid_body(self, client):
        response = client.post(
            "/forecast/coordinates/",
            json={"lat": 91, "lon": -122.4194}
        )
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_forecast_by_location_post(self, client, mock_geocoding_service, mock_weather_service):
        with patch('main.get_geocoding_service', return_value=mock_geocoding_service), \
             patch('main.get_weather_service', return_value=mock_weather_service):
            response = client.post(
                "/forecast/location/",
                json={"city": "San Francisco", "state": "CA"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["location"] == "San Francisco, CA"
            assert len(data["periods"]) == 1
            assert data["periods"][0]["temperature"] == 55
    
    @pytest.mark.asyncio
    async def test_forecast_by_location_post_invalid_body(self, client):
        response = client.post(
            "/forecast/location/",
            json={"city": "", "state": "CA"}
        )
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_geocoding_error_propagation(self, client, mock_geocoding_service):
        mock_geocoding_service.geocode.side_effect = HTTPException(
            status_code=404, 
            detail="Location not found for NonExistent, XX"
        )
        
        with patch('main.get_geocoding_service', return_value=mock_geocoding_service):
            response = client.get("/forecast/location/?city=NonExistent&state=XX")
            
            assert response.status_code == 404
            assert response.json()["detail"] == "Location not found for NonExistent, XX"
    
    @pytest.mark.asyncio
    async def test_weather_service_error_propagation(self, client, mock_weather_service):
        mock_weather_service.get_grid_point.side_effect = HTTPException(
            status_code=503,
            detail="Unable to connect to weather service, please try again later"
        )
        
        with patch('main.get_weather_service', return_value=mock_weather_service):
            response = client.get("/forecast/coordinates/?lat=37.7749&lon=-122.4194")
            
            assert response.status_code == 503
            assert "Unable to connect to weather service" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main(["-v", "test_main.py"])