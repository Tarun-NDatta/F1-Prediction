# integration_setup.py - Run this to set up the complete live race integration

import os
import django
from django.conf import settings
from django.core.management import execute_from_command_line

def setup_integration():
    """Setup script to integrate all components"""
    
    print("🏎️  F1 Live Race ML Integration Setup")
    print("=" * 50)
    
    # 1. Check environment
    print("1. Checking environment...")
    required_vars = ['RAPIDAPI_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set RAPIDAPI_KEY in your environment")
        return False
    else:
        print("✅ Environment variables OK")
    
    # 2. Database migrations
    print("\n2. Running database migrations...")
    try:
        execute_from_command_line(['manage.py', 'makemigrations'])
        execute_from_command_line(['manage.py', 'migrate'])
        print("✅ Database migrations completed")
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False
    
    # 3. Create cache table (if using database cache)
    print("\n3. Setting up cache...")
    try:
        execute_from_command_line(['manage.py', 'createcachetable'])
        print("✅ Cache table created")
    except Exception as e:
        print(f"⚠️  Cache table might already exist: {e}")
    
    # 4. Test API connection
    print("\n4. Testing HypRace API connection...")
    try:
        execute_from_command_line(['manage.py', 'run_live_predictions', '--test-connection'])
        print("✅ API connection test passed")
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    # 5. Check quota status
    print("\n5. Checking API quota...")
    try:
        execute_from_command_line(['manage.py', 'run_live_predictions', '--quota-status'])
        print("✅ API quota checked")
    except Exception as e:
        print(f"❌ Quota check failed: {e}")
    
    print("\n🎉 Integration setup completed!")
    print("\nNext steps:")
    print("1. Start your Django development server: python manage.py runserver")
    print("2. Visit /live-updates/ in your browser")
    print("3. Click 'Start Live' button to begin live race data collection")
    print("4. Or run the management command directly:")
    print("   python manage.py run_live_predictions --log-level INFO")
    
    return True

# Quick test script
def test_integration():
    """Test the integration components"""
    
    print("🧪 Testing Live Race Integration")
    print("=" * 40)
    
    # Test 1: Model import
    print("1. Testing model import...")
    try:
        from data.models import CatBoostPrediction
        print(f"✅ CatBoostPrediction model imported")
        
        # Check if table exists
        count = CatBoostPrediction.objects.count()
        print(f"✅ Database table accessible ({count} existing records)")
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    # Test 2: Cache system
    print("\n2. Testing cache system...")
    try:
        from django.core.cache import cache
        cache.set('test_key', 'test_value', 60)
        value = cache.get('test_key')
        if value == 'test_value':
            print("✅ Cache system working")
            cache.delete('test_key')
        else:
            print("❌ Cache system not working properly")
            return False
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False
    
    # Test 3: API client
    print("\n3. Testing API client...")
    try:
        from prediction.management.commands.run_live_predictions import HypRaceAPIClient
        client = HypRaceAPIClient()
        success, message = client.test_connection()
        if success:
            print(f"✅ API client: {message}")
        else:
            print(f"❌ API client: {message}")
            return False
    except Exception as e:
        print(f"❌ API client test failed: {e}")
        return False
    
    # Test 4: Views
    print("\n4. Testing views...")
    try:
        from django.test import Client
        client = Client()
        
        # Test live race data endpoint
        response = client.get('/api/live-race-data/')
        if response.status_code == 200:
            print("✅ Live race data endpoint accessible")
        else:
            print(f"❌ Live race data endpoint returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Views test failed: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    return True

# Usage instructions
USAGE_INSTRUCTIONS = """
🚀 F1 Live Race ML System - Usage Guide
=======================================

1. SETUP:
   - Run: python integration_setup.py
   - Ensure RAPIDAPI_KEY environment variable is set
   - Start Django server: python manage.py runserver

2. FRONTEND USAGE:
   - Visit: http://localhost:8000/live-updates/
   - Click "Start Live" button to begin real-time data collection
   - Watch live commentary and ML predictions update automatically
   - Use "Stop Live" to end data collection

3. COMMAND LINE USAGE:
   - Start: python manage.py run_live_predictions
   - Options:
     --test-connection    : Test API without using quota
     --quota-status      : Check remaining API calls
     --list-races        : Show available races
     --dry-run          : Run without saving to database
     --log-level INFO   : Set logging level

4. DATABASE INTEGRATION:
   - Predictions automatically saved to CatBoostPrediction table
   - View via Django admin or custom queries
   - Data includes confidence scores, position changes, timestamps

5. MONITORING:
   - Check logs: tail -f hyprace_prediction.log
   - Monitor API usage: python manage.py run_live_predictions --quota-status
   - Database queries: CatBoostPrediction.objects.filter(created_at__gte=today)

6. TROUBLESHOOTING:
   - Cache issues: python manage.py createcachetable
   - API errors: Check RAPIDAPI_KEY and quota
   - Database errors: python manage.py migrate
   - Frontend issues: Check browser console and Django logs

For support, check the logs and ensure all dependencies are installed.
"""

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'setup':
            setup_integration()
        elif sys.argv[1] == 'test':
            test_integration()
        elif sys.argv[1] == 'help':
            print(USAGE_INSTRUCTIONS)
    else:
        print("Usage: python integration_setup.py [setup|test|help]")
        print("Run 'python integration_setup.py help' for detailed instructions")