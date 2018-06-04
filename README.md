## My Project in Text Summarization

Text summarization solves the problem of condensing information into a more compact form, while maintaining the important information in the text. The methods of automatic text summarization fall into two primary categories: extractive and abstractive. A common approach of extractive summarization involves selecting the most representative sentences that best cover the information expressed by the original text based on a ranking of sentences by relevance. A popular method of abstractive text summarization is using an encoder-decoder structure, which generates a latent factor representation of the data, and decodes it to generate a summary. The goal of the project was to analyze and compare the effectiveness of both methods when applied specifically to scientific texts.

### Motivation

My motivation for this project came from personal experience. As a student in college, I'm often faced with a large number of scientific papers and research articles that pertain to my interests, yet I don't have the time to read them all. I wanted a way to be able to get summaries of the main ideas for the papers, without significant loss of important content. Text summarization is a widely implemented algorithm, but I wanted to explore different text summarization methods applied to scientific writing in particular. 

### Introduction

Automatic text summarization is the process of shortening a text documentation using a system for prioritizing information. Technologies that generate summaries take into account variables such as length, style, and syntax. Text summarization from the perspective of humans is taking a chunk of information and extracting what one deems most important. Automatic text summarization is based on the logical quantification of features of the text including, weighting keywords, and sentence ranking.

#### Extractive Text Summarization
Extractive text summarization does not use words aside from the ones already in the text, and selects some combination of the existing words most relevant to the meaning of the source. Techniques of extractive summarization include ranking sentences and phrases in order of importance and selecting the most important components of the document to construct the summary. These methods tend to more robust because they use existing phrases, but lack flexibility since they cannot use new words or paraphrase.

#### Abstractive Text Summarization
Abstractive text summarization involves generating entirely new phrases and sentences to capture the meaning of the text. Abstractive methods tend to be more complex, because the machine must read over the text and deem certain concepts to be important, and then learn to construct some cohesive phrasing of the relevant concepts. Abstractive summarization is most similar to how humans summarize, as humans often summarize by paraphrasing. 

### Materials and Methods
Although the primary goal of my project was to be able to summarize entire scientific papers, and essentially create abstracts given papers, a paper was too long of an input text to start with. I decided to first work with generating summaries given abstracts, which are much shorter than entire papers. Essentially, my project can be thought of as generating paper titles, given abstracts. First, I needed a dataset of abstract texts with their corresponding titles.

I used the NSF Research Award Abstracts 1990-2003 Data Set from the UCI machine learning repository. The dataset consisted of abstracts that had won the NSF research awards from 1990 to 2003, along with the title of the paper. For my abstractive learning, the training input X was the abstract and the title was the training input Y.

For extractive summarization, I used the TextRank algorithm, which is based on Google’s PageRank algorithm. TextRanks works by transforming the text into a graph. It regards words as vertices and the relation between words in phrases or sentences as edges. Each edge also has different weight. When one vertex links to another one, it is basically casting a vote of importance for that vertex. The importance of the vertex also dictates how heavily weighted its votes are. TextRank uses the structure of the text and the known parts of speech for words to assign a score to words that are keywords for the text.


![alt text](https://raw.githubusercontent.com/mzhao98/shiny-barnacle/blob/master/algo1.png)

### Code for Reduction

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](https://github.com/adamfabish/Reduction)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Conclusions
Text


### Contact

Michelle Zhao
mzhao@caltech.edu
