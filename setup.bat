@echo off
echo Setting up data file for Earthquake Risk Analysis application...

REM Create src directory if it doesn't exist
if not exist src mkdir src

REM Check if the data file is in the current src directory 
if exist src\earthquake_cleandata_posteda.csv (
    echo Data file already exists in src directory.
) else (
    REM Look for the data file in various locations
    if exist ..\src\earthquake_cleandata_posteda.csv (
        echo Found data file in parent src directory. Copying to local src directory...
        copy ..\src\earthquake_cleandata_posteda.csv src\
        echo File copied successfully!
    ) else if exist earthquake_cleandata_posteda.csv (
        echo Found data file in current directory. Moving to src directory...
        copy earthquake_cleandata_posteda.csv src\
        echo File copied successfully!
    ) else if exist ..\earthquake_cleandata_posteda.csv (
        echo Found data file in parent directory. Copying to src directory...
        copy ..\earthquake_cleandata_posteda.csv src\
        echo File copied successfully!
    ) else (
        echo WARNING: Data file not found in any expected location.
        echo Please manually ensure earthquake_cleandata_posteda.csv is in the src directory.
    )
)

REM Check if the file is now in the correct location
if exist src\earthquake_cleandata_posteda.csv (
    echo Data file is ready for the application.
) else (
    echo WARNING: Data file not found in any expected location.
    echo Please manually ensure earthquake_cleandata_posteda.csv is in the src directory.
)

echo Setup complete! 