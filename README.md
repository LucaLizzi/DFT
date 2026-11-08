# Setup instructions
Create and activate a virtual environment, then install the project dependencies:

- Windows PowerShell: `py -3.13 -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `python3 -m venv .venv`, then `source .venv/bin/activate`
- All platforms: `python -m pip install -r requirements.txt`

In VS Code, select the `.venv` interpreter. Run the DFT experiment with `python main2.py`.


# Audio Dataset Description
All three audio files were recorded using a smartphone microphone capturing an unamplified electric guitar
*test1_notes*: Contains individual notes A2 ($110.00\text{ Hz}$), A3 ($220.00\text{ Hz}$), D2 ($73.42\text{ Hz}$), and D3 ($146.83\text{ Hz}$).\
*test2_notes*: Contains individual notes E2 ($82.41\text{ Hz}$), D2 ($73.42\text{ Hz}$), and B3 ($246.94\text{ Hz}$).\
*test3_notes*: Contains all six open strings played individually: E2 ($82.41\text{ Hz}$), A2 ($110.00\text{ Hz}$), D3 ($146.83\text{ Hz}$), G3 ($196.00\text{ Hz}$), B3 ($246.94\text{ Hz}$), and E4 ($329.63\text{ Hz}$).