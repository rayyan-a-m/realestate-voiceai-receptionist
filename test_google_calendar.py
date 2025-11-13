import unittest
import datetime
import os
import re
from google_calendar import find_available_slots, book_appointment, get_calendar_service

class TestGoogleCalendar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure credentials are in place before running tests."""
        if not os.path.exists("credentials.json"):
            raise FileNotFoundError("credentials.json not found. Please follow GOOGLE_API_SETUP.md.")
        # Running get_calendar_service once to ensure token.json is created/refreshed.
        get_calendar_service()

    def test_01_get_calendar_service(self):
        """Test if we can get the Google Calendar service without errors."""
        print("Running test: test_get_calendar_service")
        try:
            service = get_calendar_service()
            self.assertIsNotNone(service, "Failed to get calendar service.")
            print("Success: Google Calendar service obtained.")
        except Exception as e:
            self.fail(f"get_calendar_service() raised an exception: {e}")

    def test_02_find_available_slots(self):
        """Test the find_available_slots function to ensure it returns slots or a 'no slots' message."""
        print("Running test: test_find_available_slots")
        try:
            result = find_available_slots()
            print(f"Result from find_available_slots: {result}")
            self.assertIsInstance(result, str)
            self.assertNotIn("Sorry, I encountered an error", result, "Function returned an error message.")
            print("Success: find_available_slots executed without internal errors.")
        except Exception as e:
            self.fail(f"find_available_slots() raised an unexpected exception: {e}")

    def test_03_book_and_delete_appointment(self):
        """
        Tests the full cycle of booking and then deleting an appointment.
        This is a real integration test that will create and then remove an event in your primary Google Calendar.
        """
        print("Running test: test_book_and_delete_appointment")
        
        # Step 1: Find an available slot to book
        slots_str = find_available_slots()
        if "No available slots" in slots_str:
            self.skipTest("Skipping booking test: No available slots found in the next 7 days.")
            return

        # Extract the first available slot from the string
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', slots_str)
        self.assertIsNotNone(match, "Could not parse a valid datetime from find_available_slots output.")
        test_slot = match.group(1)
        
        print(f"Attempting to book a test appointment in the first available slot: {test_slot}")

        # Step 2: Book the appointment
        full_name = "Test User"
        email = "test.user@example.com"
        property_name = "Test Property"
        
        booking_result = book_appointment(test_slot, full_name, email, property_name)
        print(f"Booking result: {booking_result}")
        self.assertIn("Success", booking_result, "Booking function did not return a success message.")

        # Step 3: Find the event that was just created to get its ID for deletion
        service = get_calendar_service()
        start_time = datetime.datetime.strptime(test_slot, "%Y-%m-%d %H:%M").isoformat()
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=f"{start_time}Z",
            maxResults=5,
            singleEvents=True,
            orderBy='startTime',
            q=f'Property Visit: {property_name} for {full_name}'
        ).execute()
        
        created_event = events_result.get('items', [])
        self.assertGreater(len(created_event), 0, "Could not find the created event in the calendar to delete it.")
        event_id = created_event[0]['id']
        print(f"Found created event with ID: {event_id}")

        # Step 4: Delete the event to clean up the calendar
        try:
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            print(f"Success: Cleaned up by deleting test event {event_id}.")
        except Exception as e:
            self.fail(f"Failed to delete the test appointment. Manual cleanup may be required for event ID {event_id}. Error: {e}")

if __name__ == '__main__':
    unittest.main()
