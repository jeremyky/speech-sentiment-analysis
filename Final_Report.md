
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
This project aims to combine speech recognition and sentiment analysis to understand human emotions in real-time conversations. The goal of the project is to utilize state-of-the-art large language models (LLMs) for sentiment detection by analyzing transcriptions generated from speech input. Our approach leverages advanced speech recognition APIs to transcribe spoken language into text, which is then processed by sentiment analysis models such as DistilRoBERTa, and then fine-tuned on datasets like GoEmotions. The primary objective is to assess the effectiveness of these models in accurately classifying emotions from transcribed speech, providing insights into user sentiment. 

\end{abstract}

\section{Introduction}
As students in this NLP class, we aim to explore the intersection of speech recognition and sentiment analysis to enhance our understanding of how large language models (LLMs) perform in real-time scenarios. Specifically, we want to learn how effectively sentiment can be derived from speech transcriptions, and how state-of-the-art models like DistilRoBERTa handle the nuances of emotional expression in spoken language. By focusing on speech-based sentiment detection, we will gain hands-on experience in fine-tuning and evaluating pre-trained models for sentiment classification tasks, a crucial skill in the field of NLP.

This project is particularly interesting because it combines two impactful areas of NLP—speech recognition and sentiment analysis—that have widespread applications, from customer service bots to mental health assistants. Real-time emotion detection can significantly enhance the interaction between users and AI, making conversational systems more empathetic and responsive.

Our timeline for the project is as follows:

\begin{itemize}
    \item \textbf{Weeks 1-2}: Set up speech recognition APIs (Whisper) and fine-tune sentiment analysis models (DistilRoBERTa) using emotion-labeled datasets like GoEmotions.
    \item \textbf{Weeks 3-4}: Conduct initial testing of speech-to-text pipelines, ensuring accurate transcription for sentiment analysis. Begin evaluating the performance of sentiment analysis models on transcribed speech, focusing on basic metrics such as accuracy and precision.
    \item \textbf{Weeks 5-6}: Refine the sentiment detection process, improving model fine-tuning and adjusting based on feedback from initial testing. Explore more advanced sentiment metrics, including F1 score and confusion matrices, to assess model performance.
    \item \textbf{Weeks 7-8}: Investigate the integration of sentiment-driven response generation for potential chatbot implementation. Test how sentiment output can influence conversation flow in chatbots or assistive applications.
    \item \textbf{Week 9}: Finalize project, document results, and prepare for presentation. Summarize findings on the effectiveness of combining speech recognition and sentiment analysis and highlight future work possibilities, such as full chatbot integration.
\end{itemize}

By the end of this project, we expect to have a deeper understanding of how well LLMs can interpret human emotions from speech, along with practical insights into the challenges of real-time sentiment analysis.


\section{Related Work}

Existing studies on OpenAI Whisper have shown highly appealing capabilities in optimizing the transcription process. Many of these existing implementations showcase unique ways of leveraging OpenAI's Whisper AI for the transcription of audio files. For example, Whisper AI can be used in mental health research, highlighting its unique capabilities in streamlining what has traditionally been a labor-intensive process. By integrating Whisper AI, researchers can optimize transcription efficiency while minimizing errors, a significant improvement over conventional methods. What sets this article apart is its detailed, step-by-step approach to implementing a transcription pipeline specifically tailored for psychology, psychiatry, and neuroscience research (Spiller et. al, 2023). It not only covers the technical setup—such as recording, preprocessing, and post-processing audio data—but also includes practical examples in a Python environment, enabling researchers to easily replicate the process. Additionally, the discussion of privacy and Institutional Review Board (IRB) considerations underscores the ethical dimensions of using AI in sensitive research areas, making this tutorial particularly relevant for researchers seeking to incorporate advanced technology while adhering to ethical standards.




In the realm of speech recognition, Yoon et al. (2023) evaluated various speech-to-text APIs, including OpenAI Whisper, for their effectiveness in transcribing emotional speech. Their research, titled "LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition," highlighted Whisper’s superior performance in handling diverse accents and emotional intonations, which aligns with our decision to adopt Whisper for speech transcription. By leveraging Whisper, we benefit from its high accuracy and robustness in transcribing varied speech patterns, ensuring reliable input for our sentiment analysis.



Additionally, Li et al. (2023) presented an innovative approach in "Improving Speech Recognition Performance in Noisy Environments by Enhancing Lip Reading Accuracy." They proposed integrating lip-reading capabilities with speech recognition to bolster performance in noisy settings. By constructing a one-to-many mapping model between lip movements and speech, and employing cross-modal fusion techniques, their method significantly reduced the Word Error Rate (WER) in challenging acoustic environments. This aligns with our project's focus on leveraging advanced speech recognition (Whisper) and sentiment analysis (DistilRoBERTa) to improve emotion detection accuracy. Incorporating visual information, as demonstrated by Li et al., could further enhance our system's ability to accurately interpret emotions from speech, especially in noisy conditions.

\section{Problem Setup}

The goal of this project is to develop a Real-Time Speech-to-Sentiment Analysis System that accurately detects human emotions from live spoken conversations. The system takes live audio input captured via a microphone, which is then transcribed into text using advanced speech recognition API OpenAI Whisper. This transcribed text is processed by sentiment analysis models such as DistilRoBERTa, fine-tuned on the GoEmotions dataset, to classify emotions into categories like joy, sadness, and anger. The pipeline is designed to operate with minimal latency, ensuring real-time performance. The outputs include detailed emotion labels with confidence scores, aggregated sentiment insights, and real-time visual feedback, which can enhance interactions in applications like customer service bots and mental health assistants. By integrating these components, the project aims to create a responsive and empathetic AI system that effectively interprets and reacts to user emotions during live conversations.



\section{Method}
Our framework consists of three major components: speech recognition, text processing for sentiment analysis, and sentiment classification using large language models.

\begin{itemize}
\item \textbf{Speech Recognition}: We employ the OpenAI Whisper API to convert spoken language into text in real-time. This API is selected for its high accuracy and ability to handle diverse accents and speaking styles.

\item \textbf{Text Pre-processing}: The transcribed text undergoes pre-processing steps such as tokenization, normalization, and removal of any transcription errors to ensure the input quality for sentiment analysis models.

\item \textbf {Sentiment Analysis}: We utilize pre-trained DistilRoBERTa models, fine-tuned on the GoEmotions dataset, to classify the emotions expressed in the transcribed text. The GoEmotions dataset provides a comprehensive set of emotion labels, enabling nuanced sentiment detection beyond simple positive or negative classifications.

\item \textbf {Integration and Inference}: The processed text is fed into the sentiment analysis models in real-time, and the resulting emotion classifications are used to generate insights or inform response generation in potential chatbot applications.

\end{itemize}

\section{Experiment setup and evaluation}
For this project, we utilize the GoEmotions dataset, which includes 58,000 Reddit comments labeled with 27 emotion categories, to fine-tune our sentiment analysis models, DistilRoBERTa. Additionally, we use diverse speech samples from various speakers, encompassing different accents, speaking styles, and background noises, to ensure the robustness of our speech recognition component. (Figure 1)


Our evaluation protocol involves assessing the speech recognition accuracy by calculating the Word Error Rate (WER) of OpenAI Whisper API against ground truth transcripts. For sentiment classification, we measure the performance of DistilRoBERTa using accuracy, precision, recall, and F1-score, along with confusion matrices to evaluate the correct classification of each emotion category. To ensure real-time functionality, we also evaluate the system’s latency and throughput, measuring the time from audio input to sentiment output and the ability to handle multiple conversations simultaneously.
(Figure 2)


The experimental procedure begins with fine-tuning the sentiment models on the GoEmotions dataset, followed by integrating the speech recognition APIs into the pipeline. We conduct initial tests with the collected speech samples to verify transcription accuracy and then evaluate the sentiment analysis performance using the defined metrics. Based on the results, we iteratively refine the models and pipeline to address any issues with transcription errors or classification inaccuracies. Finally, we perform a comprehensive evaluation to validate the system’s effectiveness in accurately and efficiently detecting emotions in real-time conversations.



\section{Results}

Preliminary results indicate that the DistilRoBERTa model, fine-tuned on the GoEmotions dataset, achieved high accuracy in emotion classification. To accommodate slight variations in parsing human speech, we conducted extensive preparation to find a practical and efficient API for ingesting and analyzing audio inputs.

We have successfully implemented a working solution for converting human speech to text using the OpenAI Whisper API. This implementation involved several steps:
\begin{itemize}
    \item Configuring the Whisper API to ensure optimal performance for our specific use case.
    \item Selecting the appropriate model size to balance accuracy and processing time.
    \item Leveraging Whisper's configurations for multiple languages, noise robustness, and real-time or batch processing, tailored to the characteristics of the audio inputs.
\end{itemize}

To facilitate integration, we developed a pipeline that captures audio input from users, processes it through the Whisper API, and retrieves the transcribed text. Error handling was implemented to address issues such as background noise and speech interruptions, enhancing transcription reliability. With Whisper API’s robust performance in challenging audio environments, we achieved high transcription accuracy, significantly reducing the manual effort required for text conversion.

Once the speech recognition tool was set up, we trained the transcribed text for sentiment analysis using the DistilRoBERTa transformer, pre-trained on emotion detection. DistilRoBERTa supports multiple emotions simultaneously and offers a good balance of speed and accuracy due to its lightweight design.


The performance of the OpenAI Whisper API was evaluated under various conditions, and the results are summarized below:
\begin{itemize}
    \item \textbf{Baseline Word Error Rate (WER)}: 16.67\%
    \item \textbf{Conditions}:
    \begin{itemize}
        \item Normal: 16.67\%
        \item Noise: 16.67\%
        \item Speed up: 16.67\%
        \item Slow down: 26.67\%
        \item Low quality: 93.33\%
    \end{itemize}
\end{itemize}

These results indicate that the Whisper API maintains consistent accuracy for normal, noisy, and faster speech patterns. However, its performance decreases for slower speech and shows significant challenges with low-quality audio inputs.


The sentiment analysis model's performance, evaluated using standard metrics, is as follows:
\begin{itemize}
    \item \textbf{Precision}: 83.33\%
    \item \textbf{Recall}: 83.33\%
    \item \textbf{F1 Score}: 83.33\%
\end{itemize}

These balanced metrics suggest that the model is equally effective at identifying relevant emotions (precision) and capturing all instances of each emotion (recall).


The system's performance for detecting individual emotions is detailed below:
\begin{itemize}
    \item \textbf{Joy}: 96.8\%
    \item \textbf{Anger}: 98.4\%
    \item \textbf{Sadness}: 99.1\%
    \item \textbf{Fear}: 98.5\%
    \item \textbf{Surprise}: 97.7\%
\end{itemize}

These high accuracy rates across different emotions demonstrate the model's strong capability to distinguish between various emotional states with remarkable precision. Overall, these results provide a comprehensive overview of the system's performance, highlighting its strengths:
\begin{itemize}
    \item Accurate emotion detection across multiple emotion categories.
    \item Robust real-time processing capabilities.
\end{itemize}
At the same time, the system presents areas for improvement, particularly in handling low-quality audio inputs. Future refinements will aim to address these challenges, especially in the context of improving the system for scalability and practicality.


\section{Challenges and solutions}
One significant challenge we have encountered in our project is ensuring the accuracy of speech transcription across diverse audio inputs. Variations in accents, speaking speeds, and background noise can lead to transcription errors, which subsequently affect the reliability of sentiment analysis. To address this, we are incorporating a diverse set of speech samples during the training phase to make the speech recognition models more robust. Additionally, we plan to implement noise reduction and audio normalization techniques during the pre-processing stage to enhance transcription quality.

Another difficulty lies in effectively fine-tuning the sentiment analysis models, BERT, on the GoEmotions dataset to accurately capture subtle emotional nuances. The complexity of emotions and their overlapping characteristics can make precise classification challenging. To overcome this, we are experimenting with various fine-tuning strategies, such as adjusting learning rates and using cross-validation techniques to optimize model performance. Furthermore, we are exploring the use of data augmentation methods to increase the diversity of training samples, thereby improving the models' ability to distinguish between closely related emotions.

\section{Conclusion}
The Real-Time Speech-to-Sentiment Analysis System developed in this project demonstrates significant potential in accurately detecting human emotions from live spoken conversations. Our key findings include:

\begin{itemize}
    \item \textbf {High Transcription Accuracy}: The OpenAI Whisper API achieved impressive speech recognition performance, with a baseline Word Error Rate (WER) of 16.67\% for normal speech conditions. This accuracy ensures reliable text input for sentiment analysis.
    \item \textbf {Robust Sentiment Classification}: The fine-tuned BERT model, trained on the GoEmotions dataset, exhibited strong performance in emotion classification. Overall sentiment analysis metrics showed 83.33\% precision, recall, and F1 score.
    \item \textbf {Excellent Individual Emotion Detection}: The system demonstrated high accuracy in identifying specific emotions, with joy at 96.8\%, anger at 98.4\%, sadness at 99.1\%, fear at 98.5\%, and surprise at 97.7\%.
    \item \textbf {Real-Time Processing}: The integration of speech recognition and sentiment analysis components allowed for efficient real-time emotion detection, making the system suitable for applications requiring immediate feedback.    
    \item \textbf {Challenges in Diverse Audio Inputs}: While the system performed well overall, speaking speeds and background noise presented challenges in maintaining consistent accuracy across all scenarios. It would be beneficial to explore options for improving the throughput for simultaneous conversations and real time processing capabilities.
\end{itemize}

Future directions for this project could include:
\begin{itemize}
    \item \textbf{Multimodal Analysis}: Incorporating visual cues, such as facial expressions or gestures, to enhance emotion detection accuracy.
    \item \textbf{Expanded Emotion Range}: Fine-tuning the model on more diverse datasets to recognize a broader spectrum of emotions and their nuances.
    \item \textbf{Adaptive Learning}: Implementing continuous learning mechanisms to improve the system's performance over time based on user interactions and feedback.
    \item \textbf{Application-Specific Optimization}: Tailoring the system for specific use cases, such as customer service bots or mental health assistants, by incorporating domain-specific knowledge and requirements.
    \item \textbf{Latency Reduction}: Further optimizing the pipeline to minimize processing time and improve real-time performance, especially for high-volume applications.
\end{itemize}

By addressing these future directions, the Real-Time Speech-to-Sentiment Analysis System can evolve into a more robust and versatile tool for understanding and responding to human emotions in various real-world applications.

\newpage
\section{Citations} \\  
\begin{hangparas}{30pt}{1}
Spiller, T. R., Rabe, F., Ben-Zion, Z., Korem, N., Burrer, A., Homan, P., Duek, O. (2023, April 27). Efficient and accurate transcription in mental health research - A tutorial on using Whisper AI for audio file transcription. https://doi.org/10.31219/osf.io/9fue8
\end{hangparas}

\begin{hangparas}{30pt}{1} Wu, Z., Gong, Z., Ai, L., Shi, P., Donbekci, K., & Hirschberg, J. (2023). Beyond silent letters: Amplifying LLMs in emotion recognition with vocal nuances. Department of Computer Science, Columbia University.
\end{hangparas}

\begin{hangparas}{30pt}{1} Jia, B., Chen, H., Sun, Y., Zhang, M., & Zhang, M. (2023). LLM-driven multimodal opinion expression identification. Interspeech. \end{hangparas}

\begin{hangparas}{30pt}{1} Fox, J. (2024, February 15). Enhanced voice AI platform with new audio intelligence models. Deepgram. Retrieved from https://deepgram.com/learn/ai-speech-audio-intelligence-sentiment-analysis-intent-recognition-topic-detection-api
\end{hangparas}

\begin{hangparas}{30pt}{1} An, M. (2024). Voice analytics: Revolutionizing customer engagement. Observe.AI. Retrieved from https://www.observe.ai/blog/voice-analytics \end{hangparas}

\begin{hangparas}{30pt}{1} Dilmegani, C., & Alp, E. (2024, September 9). Top 7 methods for audio sentiment analysis. AI Multiple. Retrieved from https://research.aimultiple.com/audio-sentiment-analysis/\end{hangparas}

\begin{hangparas}{30pt}{1} Huang, Y., Xiao, J., Tian, K., Wu, A., & Zhang, G. (2019). Research on robustness of emotion recognition under environmental noise conditions. IEEE Access, 7, 146827–146838. https://doi.org/10.1109/ACCESS.2019.2944386 \end{hangparas}

\begin{hangparas}{30pt}{1} Zhou, K., Sisman, B., & Li, H. (2021). Limited data emotional voice conversion leveraging text-to-speech: Two-stage sequence-to-sequence training. Proceedings of INTERSPEECH 2021, 30 
August – 3 September, Brno, Czechia \end{hangparas}

\begin{hangparas}{30pt}{1} Yoon, E., Yoon, H. S., Harvill, J., Hasegawa-Johnson, M., & Yoo, C. D. (2024). LI-TTA: Language informed test-time adaptation for automatic speech recognition. Proceedings of INTERSPEECH 2024.  https://doi.org/10.48550/arXiv.2408.05769 \end{hangparas}

\begin{hangparas}{30pt}{1}  Li, D., Gao, Y., Zhu, C., Wang, Q., & Wang, R. (2023). Improving speech recognition performance in noisy environments by enhancing lip reading accuracy. Sensors, 23(4), 2053. https://doi.org/10.3390/s23042053
 \end{hangparas}

\newpage
\section{Appendix}
\begin{figure}[h!]
\centering
\includegraphics[width=\linewidth]{ICLR 2025 Template/nlp.drawio.png}
\caption{Diagram of Transformer-Based Speech to Text Pipeline.}
\label{fig:nlp_pipeline}
\end{figure}

\begin{figure}[h!]
\centering
\includegraphics[width=\linewidth]{ICLR 2025 Template/nlp_final_project_diagram.png}
\caption{Final Diagram of Key Components in Sentiment Analysis Design.}
\label{fig:nlp_final_project_diagram}
\end{figure}
\end{document}
