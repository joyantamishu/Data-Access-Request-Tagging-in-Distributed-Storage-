# Data-Access-Request-Tagging-in-Distributed-Storage-

Data access request tagging is an import aspect in prefetching, cache optimization and QOS (quality of service) management 
in storage system. In a disaggregated storage system the block layer has no clue regarding the qos of storage I/O request, since no
application specific tagging is not present in the access request. So somehow it needs to classify each I/O request, or group of I/O
request in order to provide the expected quality of services.
<br/>
<br/>
The main goal of this small project was to classify each of the request, by observing the data access pattern. I have used both the naive based approach
and recurrent neural network (RNN) for the classification and found RNN better in term of accuracy. Please refer to the *Project Details.pdf*
file for more details. I tried my best to write the pdf without using any technical jargon.
<br/>
<br>
I worked on trace file found here (http://iotta.snia.org/tracetypes/3). Please note that I have not attached the trace files in
this repository. So please download the trace files before proceeding.
<br/>
<br/>
**preprocessing.py** is used to preprocess the downloaded file. Change the file name at line no 10, to make it work (default value I 
put is madmax). To change the *flowlet* threshold value (please refer to the pdf), change at line no 34.
<br/>
**bayesian_apply.py** and **bayesian_apply.py** shows the accruacy using bayesian and LSTM approach. 
<br/>
**plot_for_report.py** to see the graph.
