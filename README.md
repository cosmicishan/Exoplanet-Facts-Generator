# Exoplanet Fact Generator

This repository contains a collection of blogs focused on exoplanet research, discoveries, and related information. It also includes a Random Fact Generation (RAG) pipeline, which can generate a random fact from the list of blogs.

## Project Structure

- `data/`: Contains PDF files with extracted blog content. These PDFs include detailed information about various exoplanet discoveries and research.
  
- `fact.py`: This Python script contains the RAG (Random Fact Generation) pipeline. It generates random facts by selecting a random blog from the list of available blog posts.

- `*.ipynb`: These Jupyter notebooks are used for testing various functionalities and ensuring the correct working of the code, especially the random fact generation logic.

## How the RAG Pipeline Works (in `fact.py`)

In `fact.py`, the RAG pipeline is created to generate a random fact. The process works as follows:

1. **Blog Data**: The blog posts are stored in a list, with each post representing a different fact or piece of information.
  
2. **Random Selection**: The script selects a random blog post from this list using Python's `random` module.

3. **Fact Generation**: The selected blog post is then processed to display the random fact.

### Key Functions in `fact.py`:
- `generate_random_fact()`: This function selects a random blog from the list and returns it as a random fact.
  
- The list of blogs can be customized by adding more blog content to the predefined list.

## Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/exoplanet-research-blog.git
   ```

2. Install any necessary dependencies (if applicable):
   ```bash
   pip install -r requirements.txt
   ```

3. To generate a random fact, simply run the `fact.py` script:
   ```bash
   python fact.py
   ```

4. For testing and experimentation, you can use the Jupyter notebooks located in the repository. Run them with:
   ```bash
   jupyter notebook
   ```

## Contributing

Feel free to contribute by opening issues or submitting pull requests. Contributions are welcome!
