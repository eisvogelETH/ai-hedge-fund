from datetime import datetime


# Set default dates
end_date = datetime.now().strftime('%Y-%m-%d')

end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

# Handle the calculation of start_date based on end_date_obj
if end_date_obj.month > 3:
    # If month is greater than 3, just subtract 3 months
    start_month = end_date_obj.month - 3
    start_year = end_date_obj.year
else:
    # If month is less than or equal to 3, subtract 1 year and add 9 months
    start_month = end_date_obj.month + 9
    start_year = end_date_obj.year - 1

# Try creating the start date object safely
try:
    start_date_obj = end_date_obj.replace(year=start_year, month=start_month)
except ValueError:
    # Handle cases where the month is invalid (e.g., February 30th)
    print(f"Error: Invalid date created for {start_month}/{start_year}. Adjusting...")
    # Adjust the day if necessary to avoid "day out of range" errors
    start_date_obj = end_date_obj.replace(year=start_year, month=start_month, day=1)

# Convert the start date object to string
start_date = start_date_obj.strftime('%Y-%m-%d')

# Print results
print("End Date:", end_date)
print("Start Date:", start_date)