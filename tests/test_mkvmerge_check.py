"""
Unit tests for mkvmerge availability check in m3u8-get.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
import pytest

# Import the module
spec = importlib.util.spec_from_file_location("m3u8_get", Path(__file__).parent.parent / "m3u8-get.py")
if spec is None:
    raise ImportError("Failed to load m3u8-get module spec")
m3u8_get = importlib.util.module_from_spec(spec)
if m3u8_get is None:
    raise ImportError("Failed to create m3u8-get module")
if spec.loader is None:
    raise ImportError("Failed to get loader for m3u8-get")
spec.loader.exec_module(m3u8_get)

prompt_and_run_mkvmerge = m3u8_get.prompt_and_run_mkvmerge


class TestMkvMergeCheck:
    """Tests for mkvmerge availability check."""

    @patch("subprocess.run")
    @patch("questionary.confirm")
    @patch("builtins.print")
    def test_mkvmerge_not_found(self, mock_print, mock_confirm, mock_subprocess_run):
        """Test that missing mkvmerge is handled gracefully."""
        # Mock questionary to return True (user confirms execution)
        # We need to mock the object returned by confirm().ask()
        mock_questionary_obj = MagicMock()
        mock_questionary_obj.ask.return_value = True
        mock_confirm.return_value = mock_questionary_obj

        # Mock subprocess.run to raise FileNotFoundError
        mock_subprocess_run.side_effect = FileNotFoundError 

        # Call the function
        downloaded_files = ["file1.ts", "file2.ts"]
        output_folder = "downloads"
        file_out_name = "output"
        
        prompt_and_run_mkvmerge(downloaded_files, output_folder, file_out_name)

        # Verify that the specific error message was printed
        # We check if any of the print calls contained the error message
        found_error_message = False
        for call in mock_print.call_args_list:
            args, _ = call
            if args and "Error: 'mkvmerge' not found" in args[0]:
                found_error_message = True
                break
        
        assert found_error_message, "Error message for missing mkvmerge was not printed"
        
        # Verify that the manual command was also printed (checking for cmd_display usage in print)
        # The exact command string construction in the test is tricky to match exactly 
        # without duplicating logic, but we can check if a message starting with "👉" was printed
        found_manual_instruction = False
        for call in mock_print.call_args_list:
            args, _ = call
            if args and "You can run the command manually" in args[0]:
                found_manual_instruction = True
                break
                
        assert found_manual_instruction, "Manual command instruction was not printed"
