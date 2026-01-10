"""
File validation module for CowNet AI data uploads.

Provides validation logic for uploaded CSV files to ensure they meet
the required schema before being moved to the data directory.
"""

import os
import re
import shutil
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime


class FileType(str, Enum):
    """Supported file types for upload."""
    COW_LOCATION = "cow_location"
    COW_REGISTRY = "cow_registry"
    PEN_ASSIGNMENT = "pen_assignment"


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    file_type: FileType
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "file_type": self.file_type.value,
            "errors": self.errors,
            "warnings": self.warnings
        }


def is_valid_uuid(value) -> bool:
    """Check if a value is a valid UUID string."""
    if pd.isna(value):
        return False
    uuid_pattern = re.compile(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    )
    return bool(uuid_pattern.match(str(value)))


def is_valid_int(value) -> bool:
    """Check if a value can be interpreted as an integer."""
    if pd.isna(value):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_uuid_or_int(value) -> bool:
    """Check if a value is either a valid UUID or an integer."""
    return is_valid_uuid(value) or is_valid_int(value)


def is_valid_unix_timestamp(value) -> bool:
    """Check if a value is a valid UNIX timestamp."""
    if pd.isna(value):
        return False
    try:
        ts = float(value)
        # Reasonable timestamp range (1970 to 2100)
        return 0 <= ts <= 4102444800
    except (ValueError, TypeError):
        return False


def is_valid_float(value) -> bool:
    """Check if a value can be interpreted as a float."""
    if pd.isna(value):
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_iso8601_week(value) -> bool:
    """
    Check if a value is a valid ISO-8601 week format (YYYY-Www).
    Examples: 2024-W01, 2024-W52
    """
    if pd.isna(value):
        return False
    pattern = re.compile(r'^\d{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$')
    return bool(pattern.match(str(value)))


def is_valid_string(value) -> bool:
    """Check if a value is a non-empty string."""
    if pd.isna(value):
        return False
    return isinstance(value, str) and len(str(value).strip()) > 0


def validate_csv_file(file_path: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Validate that a file is a valid CSV.
    
    Returns:
        Tuple of (is_valid, dataframe or None, error_message)
    """
    if not file_path.lower().endswith('.csv'):
        return False, None, "File must have .csv extension"
    
    if not os.path.exists(file_path):
        return False, None, f"File not found: {file_path}"
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return False, None, "CSV file is empty"
        return True, df, ""
    except pd.errors.EmptyDataError:
        return False, None, "CSV file is empty or has no valid data"
    except pd.errors.ParserError as e:
        return False, None, f"Invalid CSV format: {str(e)}"
    except Exception as e:
        return False, None, f"Error reading CSV: {str(e)}"


def validate_cow_location(df: pd.DataFrame) -> ValidationResult:
    """
    Validate cow location file schema.
    
    Required columns (datatype):
    - cow_id (uuid/int)
    - timestamp (UNIX)
    - x_coor (float)
    - y_coor (float)
    - z_coor (float)
    """
    errors = []
    warnings = []
    
    required_columns = ['cow_id', 'timestamp', 'x_coor', 'y_coor', 'z_coor']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return ValidationResult(
            is_valid=False,
            file_type=FileType.COW_LOCATION,
            errors=errors,
            warnings=warnings
        )
    
    # Validate cow_id (uuid or int)
    invalid_cow_ids = []
    for idx, val in df['cow_id'].items():
        if not is_valid_uuid_or_int(val):
            invalid_cow_ids.append(idx)
    if invalid_cow_ids:
        errors.append(f"Invalid cow_id values at rows: {invalid_cow_ids[:10]}{'...' if len(invalid_cow_ids) > 10 else ''}")
    
    # Validate timestamp (UNIX)
    invalid_timestamps = []
    for idx, val in df['timestamp'].items():
        if not is_valid_unix_timestamp(val):
            invalid_timestamps.append(idx)
    if invalid_timestamps:
        errors.append(f"Invalid timestamp values at rows: {invalid_timestamps[:10]}{'...' if len(invalid_timestamps) > 10 else ''}")
    
    # Validate coordinates (float)
    for coord in ['x_coor', 'y_coor', 'z_coor']:
        invalid_coords = []
        for idx, val in df[coord].items():
            if not is_valid_float(val):
                invalid_coords.append(idx)
        if invalid_coords:
            errors.append(f"Invalid {coord} values at rows: {invalid_coords[:10]}{'...' if len(invalid_coords) > 10 else ''}")
    
    # Add warnings for potential issues
    if df.duplicated(subset=['cow_id', 'timestamp']).any():
        warnings.append("Duplicate cow_id + timestamp combinations detected")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        file_type=FileType.COW_LOCATION,
        errors=errors,
        warnings=warnings
    )


def validate_cow_registry(df: pd.DataFrame) -> ValidationResult:
    """
    Validate cow registry file schema.
    
    Required columns (datatype):
    - cow_id (uuid/int)
    - parity (int)
    - lactation_stage (string)
    - week_id (ISO-8601)
    """
    errors = []
    warnings = []
    
    required_columns = ['cow_id', 'parity', 'lactation_stage', 'week_id']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return ValidationResult(
            is_valid=False,
            file_type=FileType.COW_REGISTRY,
            errors=errors,
            warnings=warnings
        )
    
    # Validate cow_id (uuid or int)
    invalid_cow_ids = []
    for idx, val in df['cow_id'].items():
        if not is_valid_uuid_or_int(val):
            invalid_cow_ids.append(idx)
    if invalid_cow_ids:
        errors.append(f"Invalid cow_id values at rows: {invalid_cow_ids[:10]}{'...' if len(invalid_cow_ids) > 10 else ''}")
    
    # Validate parity (int)
    invalid_parity = []
    for idx, val in df['parity'].items():
        if not is_valid_int(val):
            invalid_parity.append(idx)
    if invalid_parity:
        errors.append(f"Invalid parity values at rows: {invalid_parity[:10]}{'...' if len(invalid_parity) > 10 else ''}")
    
    # Validate lactation_stage (string)
    invalid_lactation = []
    for idx, val in df['lactation_stage'].items():
        if not is_valid_string(val):
            invalid_lactation.append(idx)
    if invalid_lactation:
        errors.append(f"Invalid lactation_stage values at rows: {invalid_lactation[:10]}{'...' if len(invalid_lactation) > 10 else ''}")
    
    # Validate week_id (ISO-8601)
    invalid_weeks = []
    for idx, val in df['week_id'].items():
        if not is_valid_iso8601_week(val):
            invalid_weeks.append(idx)
    if invalid_weeks:
        errors.append(f"Invalid week_id values (expected ISO-8601 format YYYY-Www) at rows: {invalid_weeks[:10]}{'...' if len(invalid_weeks) > 10 else ''}")
    
    # Add warnings for potential issues
    if df.duplicated(subset=['cow_id', 'week_id']).any():
        warnings.append("Duplicate cow_id + week_id combinations detected")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        file_type=FileType.COW_REGISTRY,
        errors=errors,
        warnings=warnings
    )


def validate_pen_assignment(df: pd.DataFrame) -> ValidationResult:
    """
    Validate pen assignment file schema.
    
    Required columns (datatype):
    - cow_id (uuid/int)
    - pen_id (int)
    - week_id (ISO-8601)
    """
    errors = []
    warnings = []
    
    required_columns = ['cow_id', 'pen_id', 'week_id']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return ValidationResult(
            is_valid=False,
            file_type=FileType.PEN_ASSIGNMENT,
            errors=errors,
            warnings=warnings
        )
    
    # Validate cow_id (uuid or int)
    invalid_cow_ids = []
    for idx, val in df['cow_id'].items():
        if not is_valid_uuid_or_int(val):
            invalid_cow_ids.append(idx)
    if invalid_cow_ids:
        errors.append(f"Invalid cow_id values at rows: {invalid_cow_ids[:10]}{'...' if len(invalid_cow_ids) > 10 else ''}")
    
    # Validate pen_id (int)
    invalid_pen_ids = []
    for idx, val in df['pen_id'].items():
        if not is_valid_int(val):
            invalid_pen_ids.append(idx)
    if invalid_pen_ids:
        errors.append(f"Invalid pen_id values at rows: {invalid_pen_ids[:10]}{'...' if len(invalid_pen_ids) > 10 else ''}")
    
    # Validate week_id (ISO-8601)
    invalid_weeks = []
    for idx, val in df['week_id'].items():
        if not is_valid_iso8601_week(val):
            invalid_weeks.append(idx)
    if invalid_weeks:
        errors.append(f"Invalid week_id values (expected ISO-8601 format YYYY-Www) at rows: {invalid_weeks[:10]}{'...' if len(invalid_weeks) > 10 else ''}")
    
    # Add warnings for potential issues
    if df.duplicated(subset=['cow_id', 'week_id']).any():
        warnings.append("Duplicate cow_id + week_id combinations detected - cow may be assigned to multiple pens")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        file_type=FileType.PEN_ASSIGNMENT,
        errors=errors,
        warnings=warnings
    )


def validate_file(file_path: str, file_type: FileType) -> ValidationResult:
    """
    Main validation function that validates a file based on its type.
    
    Args:
        file_path: Path to the file to validate
        file_type: Type of file to validate
        
    Returns:
        ValidationResult containing validation status and any errors/warnings
    """
    # First validate it's a valid CSV
    is_valid_csv, df, error_msg = validate_csv_file(file_path)
    if not is_valid_csv:
        return ValidationResult(
            is_valid=False,
            file_type=file_type,
            errors=[error_msg],
            warnings=[]
        )
    
    # Validate based on file type
    validators = {
        FileType.COW_LOCATION: validate_cow_location,
        FileType.COW_REGISTRY: validate_cow_registry,
        FileType.PEN_ASSIGNMENT: validate_pen_assignment
    }
    
    validator = validators.get(file_type)
    if not validator:
        return ValidationResult(
            is_valid=False,
            file_type=file_type,
            errors=[f"Unknown file type: {file_type}"],
            warnings=[]
        )
    
    return validator(df)


def move_validated_file(
    source_path: str,
    destination_dir: str,
    file_type: FileType
) -> Tuple[bool, str, str]:
    """
    Move a validated file from temporary storage to the data directory.
    
    Args:
        source_path: Path to the validated file in temp storage
        destination_dir: Path to the destination data directory
        file_type: Type of file being moved
        
    Returns:
        Tuple of (success, destination_path, error_message)
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Generate destination filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(source_path)
        name_without_ext = os.path.splitext(original_name)[0]
        new_filename = f"{file_type.value}_{timestamp}.csv"
        
        destination_path = os.path.join(destination_dir, new_filename)
        
        # Move the file
        shutil.move(source_path, destination_path)
        
        return True, destination_path, ""
        
    except Exception as e:
        return False, "", f"Failed to move file: {str(e)}"


def cleanup_temp_file(file_path: str) -> bool:
    """
    Remove a file from temporary storage (e.g., after failed validation).
    
    Args:
        file_path: Path to the file to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception:
        return False


# ============================================================================
# CLI for standalone validation
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CowNet data files")
    parser.add_argument("file_path", help="Path to the CSV file to validate")
    parser.add_argument(
        "file_type",
        choices=["cow_location", "cow_registry", "pen_assignment"],
        help="Type of file to validate"
    )
    
    args = parser.parse_args()
    
    file_type = FileType(args.file_type)
    result = validate_file(args.file_path, file_type)
    
    print(f"\nValidation Result for: {args.file_path}")
    print(f"File Type: {result.file_type.value}")
    print(f"Valid: {result.is_valid}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    exit(0 if result.is_valid else 1)
