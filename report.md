
\documentclass{article} % For LaTeX2e
\usepackage{iclr2025_conference,times}

\usepackage{graphicx}

% Optional math commands from https://github.com/goodfeli/dlbook_notation.
\input{math_commands.tex}

\usepackage{hanging}
\usepackage{hyperref}
\usepackage{url}


\title{Real-Time Speech-to-Sentiment: \\
Speech Analysis Using LLMs}

% Authors must not appear in the submitted version. They should be hidden
% as long as the \iclrfinalcopy macro remains commented out below.
% Non-anonymous submissions will be rejected without review.

\author{Aaron Park, Jeremy Ky, Davis Wang \\
\texttt{\{ync4hn,juh7hc,bqe6ue\}@virginia.edu}
}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to \LaTeX{} to determine where to break
% the lines. Using \AND forces a linebreak at that point. So, if \LaTeX{}
% puts 3 of 4 authors names on the first line, and the last on the second
% line, try using \AND instead of \And before the third author name.

\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}

\iclrfinalcopy % Uncomment for camera-ready version, but NOT for submission.
\begin{document}


\maketitle

\begin{abstract}
% The abstract paragraph should be indented 1/2~inch (3~picas) on both left and
% right-hand margins. Use 10~point type, with a vertical spacing of 11~points.
% The word \textsc{Abstract} must be centered, in small caps, and in point size 12. Two
% line spaces precede the abstract. The abstract must be limited to one
% paragraph.
This project aims to combine speech recognition and sentiment analysis to understand human emotions in real-time conversations. The goal of the project is to utilize state-of-the-art large language models (LLMs) for sentiment detection by analyzing transcriptions generated from speech input. Our approach leverages advanced speech recognition APIs to transcribe spoken language into text, which is then processed by sentiment analysis models such as BERT, and then fine-tuned on datasets like GoEmotions. The primary objective is to assess the effectiveness of these models in accurately classifying emotions from transcribed speech, providing insights into user sentiment. 

\end{abstract}

\section{Introduction}
As students in this NLP class, we aim to explore the intersection of speech recognition and sentiment analysis to enhance our understanding of how large language models (LLMs) perform in real-time scenarios. Specifically, we want to learn how effectively sentiment can be derived from speech transcriptions, and how state-of-the-art models like BERT handle the nuances of emotional expression in spoken language. By focusing on speech-based sentiment detection, we will gain hands-on experience in fine-tuning and evaluating pre-trained models for sentiment classification tasks, a crucial skill in the field of NLP.

This project is particularly interesting because it combines two impactful areas of NLP—speech recognition and sentiment analysis—that have widespread applications, from customer service bots to mental health assistants. Real-time emotion detection can significantly enhance the interaction between users and AI, making conversational systems more empathetic and responsive.

Our timeline for the project is as follows:

\begin{itemize}
    \item \textbf{Weeks 1-2}: Set up speech recognition APIs (Whisper) and fine-tune sentiment analysis models (BERT) using emotion-labeled datasets like GoEmotions.
    \item \textbf{Weeks 3-4}: Conduct initial testing of speech-to-text pipelines, ensuring accurate transcription for sentiment analysis. Begin evaluating the performance of sentiment analysis models on transcribed speech, focusing on basic metrics such as accuracy and precision.
    \item \textbf{Weeks 5-6}: Refine the sentiment detection process, improving model fine-tuning and adjusting based on feedback from initial testing. Explore more advanced sentiment metrics, including F1 score and confusion matrices, to assess model performance.
    \item \textbf{Weeks 7-8}: Investigate the integration of sentiment-driven response generation for potential chatbot implementation. Test how sentiment output can influence conversation flow in chatbots or assistive applications.
    \item \textbf{Week 9}: Finalize project, document results, and prepare for presentation. Summarize findings on the effectiveness of combining speech recognition and sentiment analysis and highlight future work possibilities, such as full chatbot integration.
\end{itemize}

By the end of this project, we expect to have a deeper understanding of how well LLMs can interpret human emotions from speech, along with practical insights into the challenges of real-time sentiment analysis.

\section{Problem Setup}

The goal of this project is to develop a Real-Time Speech-to-Sentiment Analysis System that accurately detects human emotions from live spoken conversations. The system takes live audio input captured via a microphone, which is then transcribed into text using advanced speech recognition API OpenAI Whisper. This transcribed text is processed by sentiment analysis models such as BERT, fine-tuned on the GoEmotions dataset, to classify emotions into categories like joy, sadness, and anger. The pipeline is designed to operate with minimal latency, ensuring real-time performance. The outputs include detailed emotion labels with confidence scores, aggregated sentiment insights, and real-time visual feedback, which can enhance interactions in applications like customer service bots and mental health assistants. By integrating these components, the project aims to create a responsive and empathetic AI system that effectively interprets and reacts to user emotions during live conversations.



\section{Method}
Our framework consists of three major components: speech recognition, text processing for sentiment analysis, and sentiment classification using large language models.

\begin{itemize}
\item \textbf Speech Recognition: We employ the OpenAI Whisper API to convert spoken language into text in real-time. This API is selected for its high accuracy and ability to handle diverse accents and speaking styles.

\item \textbf Text Pre-processing: The transcribed text undergoes pre-processing steps such as tokenization, normalization, and removal of any transcription errors to ensure the input quality for sentiment analysis models.

\item \textbf Sentiment Analysis: We utilize pre-trained BERT models, fine-tuned on the GoEmotions dataset, to classify the emotions expressed in the transcribed text. The GoEmotions dataset provides a comprehensive set of emotion labels, enabling nuanced sentiment detection beyond simple positive or negative classifications.

\item \textbf Integration and Inference: The processed text is fed into the sentiment analysis models in real-time, and the resulting emotion classifications are used to generate insights or inform response generation in potential chatbot applications.

\end{itemize}
Diagram:

\includegraphics[width=\linewidth]{ICLR 2025 Template/nlp.drawio.png}

\newpage
\section{Experiment setup and evaluation}
For this project, we utilize the GoEmotions dataset, which includes 58,000 Reddit comments labeled with 27 emotion categories, to fine-tune our sentiment analysis models, BERT. Additionally, we use diverse speech samples from various speakers, encompassing different accents, speaking styles, and background noises, to ensure the robustness of our speech recognition component.

Our evaluation protocol involves assessing the speech recognition accuracy by calculating the Word Error Rate (WER) of OpenAI Whisper API against ground truth transcripts. For sentiment classification, we measure the performance of BERT using accuracy, precision, recall, and F1-score, along with confusion matrices to evaluate the correct classification of each emotion category. To ensure real-time functionality, we also evaluate the system’s latency and throughput, measuring the time from audio input to sentiment output and the ability to handle multiple conversations simultaneously.

The experimental procedure begins with fine-tuning the sentiment models on the GoEmotions dataset, followed by integrating the speech recognition APIs into the pipeline. We conduct initial tests with the collected speech samples to verify transcription accuracy and then evaluate the sentiment analysis performance using the defined metrics. Based on the results, we iteratively refine the models and pipeline to address any issues with transcription errors or classification inaccuracies. Finally, we perform a comprehensive evaluation to validate the system’s effectiveness in accurately and efficiently detecting emotions in real-time conversations.



\section{Results obtained so far}
Preliminary results indicate that the BERT model, fine-tuned on the GoEmotions dataset, achieved high accuracy in emotion classification. To accommodate slight variations in parsing human speech, we aimed to perform extensive preparation for finding a practical and efficient API for ingesting and analyzing audio inputs.

Our progress has advanced to a stage where we have successfully implemented a working solution for converting human speech to text using the OpenAI Whisper API. This implementation involves several steps: first, we configured the Whisper API to ensure optimal performance for our specific use case. We chose the appropriate model size based on our needs, balancing accuracy and processing time. The API offers various configurations, including support for multiple languages, noise robustness, and options for real-time or batch processing, which we tailored to suit the characteristics of the audio inputs we expect to handle.

To facilitate the integration, we developed a pipeline that captures audio input from users, processes it through the Whisper API, and retrieves the transcribed text. We implemented error handling to manage issues such as background noise and interruptions in speech, enhancing the reliability of the transcription. With the Whisper API's ability to perform well even in challenging audio environments, we were able to achieve high transcription accuracy, significantly reducing the manual effort required for text conversion.

Having established this critical milestone, our next direction involves training the transcribed text for sentiment analysis. We plan to employ various machine learning techniques and sentiment classification models, leveraging existing datasets and possibly fine-tuning pre-trained models on our specific data. This step will enhance our system's capability to understand and interpret the emotional nuances present in the spoken content. By providing deeper insights into the emotional context of conversations, we aim to improve the overall functionality and effectiveness of our project, ultimately leading to a more intuitive human-computer interaction experience




\section{Challenges and solutions}
One significant challenge we have encountered in our project is ensuring the accuracy of speech transcription across diverse audio inputs. Variations in accents, speaking speeds, and background noise can lead to transcription errors, which subsequently affect the reliability of sentiment analysis. To address this, we are incorporating a diverse set of speech samples during the training phase to make the speech recognition models more robust. Additionally, we plan to implement noise reduction and audio normalization techniques during the pre-processing stage to enhance transcription quality.

Another difficulty lies in effectively fine-tuning the sentiment analysis models, BERT, on the GoEmotions dataset to accurately capture subtle emotional nuances. The complexity of emotions and their overlapping characteristics can make precise classification challenging. To overcome this, we are experimenting with various fine-tuning strategies, such as adjusting learning rates and using cross-validation techniques to optimize model performance. Furthermore, we are exploring the use of data augmentation methods to increase the diversity of training samples, thereby improving the models' ability to distinguish between closely related emotions.

\section{Related Work}

Existing studies on OpenAI Whisper have shown highly appealing capabilities in optimizing the transcription process. Many of these existing implementations showcase unique ways of leveraging OpenAI's Whisper AI for the transcription of audio files. For example, Whisper AI can be used in mental health research, highlighting its unique capabilities in streamlining what has traditionally been a labor-intensive process. By integrating Whisper AI, researchers can optimize transcription efficiency while minimizing errors, a significant improvement over conventional methods. What sets this article apart is its detailed, step-by-step approach to implementing a transcription pipeline specifically tailored for psychology, psychiatry, and neuroscience research (Spiller et. al, 2023). It not only covers the technical setup—such as recording, preprocessing, and post-processing audio data—but also includes practical examples in a Python environment, enabling researchers to easily replicate the process. Additionally, the discussion of privacy and Institutional Review Board (IRB) considerations underscores the ethical dimensions of using AI in sensitive research areas, making this tutorial particularly relevant for researchers seeking to incorporate advanced technology while adhering to ethical standards.




In the realm of speech recognition, Yoon et al. (2023) evaluated various speech-to-text APIs, including OpenAI Whisper, for their effectiveness in transcribing emotional speech. Their research, titled "LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition," highlighted Whisper’s superior performance in handling diverse accents and emotional intonations, which aligns with our decision to adopt Whisper for speech transcription. By leveraging Whisper, we benefit from its high accuracy and robustness in transcribing varied speech patterns, ensuring reliable input for our sentiment analysis.



Additionally, Li et al. (2023) presented an innovative approach in "Improving Speech Recognition Performance in Noisy Environments by Enhancing Lip Reading Accuracy." They proposed integrating lip-reading capabilities with speech recognition to bolster performance in noisy settings. By constructing a one-to-many mapping model between lip movements and speech, and employing cross-modal fusion techniques, their method significantly reduced the Word Error Rate (WER) in challenging acoustic environments. This aligns with our project's focus on leveraging advanced speech recognition (Whisper) and sentiment analysis (BERT) to improve emotion detection accuracy. Incorporating visual information, as demonstrated by Li et al., could further enhance our system's ability to accurately interpret emotions from speech, especially in noisy conditions.



\newpage
Citations: \\ \\ 
\begin{hangparas}{15pt}{1}  % Adjust the indent (15pt) and line spacing (1) as needed
Spiller, T. R., Rabe, F., Ben-Zion, Z., Korem, N., Burrer, A., Homan, P., Duek, O. (2023, April 27). Efficient and accurate transcription in mental health research - A tutorial on using Whisper AI for audio file transcription. https://doi.org/10.31219/osf.io/9fue8
\end{hangparas}

\begin{itemize}
    \item \url{https://arxiv.org/html/2407.21315v1}
    \item \url{https://arxiv.org/html/2406.18088v2}
    \item \url{https://deepgram.com/learn/ai-speech-audio-intelligence-sentiment-analysis-intent-recognition-topic-detection-api}
    \item \url{https://www.observe.ai/blog/voice-analytics}
    \item \url{https://research.aimultiple.com/audio-sentiment-analysis/}
    \item \url{https://www.assemblyai.com/products/speech-understanding}
    \item \url{https://www.reddit.com/r/singularity/comments/1bpailv/you_guys_have_to_try_this_new_empathy_llm_demo_it/}
    \item \url{https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8852707}
    \item \url{https://www.isca-archive.org/interspeech_2021/zhou21b_interspeech.pdf}
    \item \url{https://arxiv.org/abs/2408.05769}
    \item \url{https://www.mdpi.com/1424-8220/23/4/2053}
\end{itemize}



\end{document}
