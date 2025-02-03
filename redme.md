# 📄 Legal Document Analysis System

Welcome to the **Legal Document Analysis System** project! This application enables you to parse legal documents, generate concise summaries, and detect potential risks in real-time. Built with advanced AI models and Streamlit, this tool provides an interactive user experience for legal professionals.

## 🚀 Features

- 📤 Upload Legal Documents:** Seamlessly upload your legal documents through an easy-to-use interface.
- 👀 Preview Documents:** View the content of your documents directly within the application.
- 📝 Summarize Documents:** Generate insightful summaries of each uploaded document using AI-powered summarization.
- ⚠️ Risk Detection:** Automatically flag potential risks and issues in contracts based on real-time regulatory updates.
- 📈 Interactive Dashboard:** Review summaries and flagged risks through an interactive dashboard.
- 🔔 Real-Time Notifications:** Receive alerts via email or Google Sheets for important updates and flagged risks.

## 🛠️ Installation

Follow these steps to set up the project locally:

1. **🔀 Clone the Repository:**
   ```bash
   git clone https://github.com/sohampawar7030/legal-document-analysis.git
   ```
2. 📥 Install Required System Dependencies:
⚠️ Python Compatibility: This project requires Python version 3.9 to 3.11 (3.10 recommended). Python 3.13 is not supported by some dependencies.
First, ensure you have the correct Python version:
```bash
python --version  # Windows
python3 --version  # macOS/Linux
```
Windows:
```bash
winget install Python.Python.3.10
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```
macOS/Linux:
```bash
brew install python@3.10
python3.10 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
```
3. 📦 Install Dependencies: Create a requirements.txt file with the following libraries:
   ```bash
   streamlit
   pandas
   PyPDF2
   mistralai
   groq-langchain  # For summarization
   llama   # For Meta LLaMA (if applicable)
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```
 4.  🔑 Configure API Access: If using OpenAI GPT or Meta LLaMA, ensure you have the necessary API keys. Set Environment Variables:

```bash
export GROQ_API_KEY="your_groq_api_key"  # macOS/Linux
 setx GROQ_API_KEY "your_groq_api_key"  # Windows
```
5. 📥 Install Additional Dependencies:
   Make sure to install any additional tools required for PDF processing:
Windows:
```bash
winget install poppler
winget install tesseract-ocr
```
macOS:
```bash
brew install poppler
brew install tesseract
```
🖥️ Usage
Run the Streamlit application using the following command:
```bash
streamlit run app.py
```
Once the application starts, follow these steps:

- 📂 Upload Your Legal Document:
Navigate to the sidebar and use the file uploader to select your legal document.
- 🔍 Preview the Document:
After uploading, the application will display a preview of your document.
- 📝 Summarize the Document:
Generate and view summaries for the uploaded document.
- ⚠️ Detect Risks:
The system will analyze the document for potential risks and provide alerts.
- 📁 Project Structure
```bash
legal-document-analysis/
├── app.py           
├── requirements.txt            
├──credentials.json
│──legal_document_analysis.py      
│──up.py      
├── README.md
|──rag_pipeline.py         
|──arial.ttf
```
- app.py: The main Streamlit application file.
- requirements.txt: Lists all the project dependencies.
 # 🧰 Dependencies
- The project relies on the following key libraries:
- Streamlit: For building the interactive web application.
- Pandas: For data manipulation and analysis.
- PyPDF2: For reading and handling PDF files.
- Mistralai: For AI-powered summarization.
- Groq: For utilizing GPT models.
- Meta LLaMA: For large language model tasks.
# 🌐 Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the Repository
2. Create a New Branch:
```bash
git checkout -b feature/YourFeature
```
3. Commit Your Changes:
```bash
git commit -m "Add your message here"
```
4. Push to the Branch:
```bash
git push origin feature/YourFeature
```
5. Open a Pull Request
# 📝 License
This project is licensed under the MIT License.

# 📧 Contact
For any inquiries or feedback, please reach out to Your Name.
```bash
Feel free to replace `yourusername`, `your.email@example.com`, and other placeholders with the appropriate information specific to your project.
```
