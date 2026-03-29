# tender-market-analytics

### Running the app

Deployed app can be found [here](https://tender-analytics-platform.streamlit.app/).

To run locally, do

```
git clone https://github.com/minhchaudo/tender-market-analytics
cd tender-market-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with `OPENAI_API_KEY=<YOUR_API_KEY_HERE>`, and run

```
streamlit run app.py
```

See help tooltips `(?)` in the app for instructions and recommendations.

### Acknowledgements

Data is retrieved from [muasamcong](https://muasamcong.mpi.gov.vn/).
