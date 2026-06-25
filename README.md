# MLLM-diagram-recognition

## Overview

We investigate the use of three open-source MLMMs (Gemma4-31B, InternVL3.5-14B, Qwen3.5-27B) for extracting structured information from graph-based diagrams (e.g., flowcharts, BPMN, etc.) under different settings.  

This repository contains the code, data, and evaluation scripts for our research. The results section provides a thorough overview of our results in tabular form. The `results/` folder contains fine-grained scores for each evaluation setting as csv files. 


## Repository Structure

- `data/labels/`: ground-truth JSON annotations for the diagram datasets used in our study. Note: Labels for SEM and CBD are not included due to licensing constraints.
- `data/labels/`: ground-truth JSON annotations for the with bboxes. 
- `evaluation/`: Evaluation scripts and MLLM extraction scripts. 
- `results/`: CSV files for our results 
- `prompts/`: Text files for the prompts we used 
- `qualitative/`: Image examples of ground-truth/predictions for the bounding box task


## Results 
The tables below show F1 scores for our evaluation results across the several settings discussed in the paper. 

### End-to-End Recognition (Baseline) (Levenshtein threshold = 0)

**F1.** Node text extracted with either PaddleOCR or Qwen3.5-4B. Node Recognition combines text and class; Edge Recognition uses endpoints. Note: Scores in the paper refer to Node Recognition for text only. 

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">PaddleOCR</th>
      <th colspan="2">Qwen</th>
    </tr>
    <tr>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>cbd</td><td><b>0.8891</b></td><td><b>0.6184</b></td><td>0.7151</td><td>0.3676</td></tr>
    <tr><td>fa</td><td>0.7059</td><td>0.4927</td><td><b>0.9632</b></td><td><b>0.8271</b></td></tr>
    <tr><td>fc_a</td><td>0.4085</td><td>0.1077</td><td><b>0.7948</b></td><td><b>0.5435</b></td></tr>
    <tr><td>hdBPMN-icdar2021</td><td>0.2049</td><td>0.0110</td><td><b>0.5793</b></td><td><b>0.2164</b></td></tr>
    <tr><td>sems</td><td>0.6761</td><td>0.4678</td><td><b>0.8058</b></td><td><b>0.5785</b></td></tr>
  </tbody>
</table>

### End-to-End Recognition (five-shot, Levenshtein threshold = 0)

**F1 ± std.** Node Recognition combines text and class; Edge Recognition combines endpoints, label, and class.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Gemma4</th>
      <th colspan="2">InternVL</th>
      <th colspan="2">Qwen</th>
    </tr>
    <tr>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>cbd</td><td><b>0.9657 ± 0.0000</b></td><td><b>0.8753 ± 0.0016</b></td><td>0.9367 ± 0.0004</td><td>0.7653 ± 0.0035</td><td>0.9354 ± 0.0051</td><td>0.8423 ± 0.0068</td></tr>
    <tr><td>fa</td><td><b>0.9865 ± 0.0000</b></td><td><b>0.8709 ± 0.0013</b></td><td>0.8596 ± 0.0036</td><td>0.6718 ± 0.0043</td><td>0.9228 ± 0.0281</td><td>0.8011 ± 0.0236</td></tr>
    <tr><td>fc_a</td><td><b>0.8365 ± 0.0017</b></td><td><b>0.6976 ± 0.0017</b></td><td>0.8287 ± 0.0019</td><td>0.5734 ± 0.0041</td><td>0.8098 ± 0.0119</td><td>0.6701 ± 0.0141</td></tr>
    <tr><td>hdBPMN-icdar2021</td><td><b>0.7716 ± 0.0024</b></td><td><b>0.3885 ± 0.0027</b></td><td>0.4767 ± 0.0033</td><td>0.1048 ± 0.0041</td><td>0.5216 ± 0.0263</td><td>0.3129 ± 0.0158</td></tr>
    <tr><td>sap-sam-bpmn</td><td>0.3967 ± 0.0052</td><td>0.2142 ± 0.0042</td><td><b>0.6227 ± 0.0022</b></td><td>0.1901 ± 0.0051</td><td>0.5788 ± 0.0569</td><td><b>0.3440 ± 0.0506</b></td></tr>
    <tr><td>sap-sam-uml</td><td><b>0.9461 ± 0.0052</b></td><td><b>0.2034 ± 0.0051</b></td><td>0.7500 ± 0.0185</td><td>0.0792 ± 0.0062</td><td>0.7327 ± 0.0378</td><td>0.1203 ± 0.0247</td></tr>
    <tr><td>sems</td><td><b>0.7291 ± 0.0026</b></td><td><b>0.6915 ± 0.0045</b></td><td>0.6373 ± 0.0039</td><td>0.3107 ± 0.0088</td><td>0.6006 ± 0.0184</td><td>0.5290 ± 0.0099</td></tr>
    <tr><td>synth_sem</td><td><b>0.7455 ± 0.0060</b></td><td><b>0.2987 ± 0.0033</b></td><td>0.5821 ± 0.0060</td><td>0.1223 ± 0.0033</td><td>0.3720 ± 0.0214</td><td>0.1552 ± 0.0054</td></tr>
    <tr><td>synth_fc</td><td><b>0.7458 ± 0.0046</b></td><td><b>0.6179 ± 0.0036</b></td><td>0.3300 ± 0.1845</td><td>0.2370 ± 0.1326</td><td>0.3611 ± 0.0342</td><td>0.2511 ± 0.0170</td></tr>
  </tbody>
</table>

### End-to-End Recognition (five-shot, Levenshtein threshold = 0.1)

**F1 ± std.** Node Recognition combines text and class; Edge Recognition combines endpoints, label, and class.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Gemma4</th>
      <th colspan="2">InternVL</th>
      <th colspan="2">Qwen</th>
    </tr>
    <tr>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>cbd</td><td><b>0.9851 ± 0.0000</b></td><td><b>0.9205 ± 0.0016</b></td><td>0.9618 ± 0.0004</td><td>0.8132 ± 0.0035</td><td>0.9569 ± 0.0049</td><td>0.8938 ± 0.0066</td></tr>
    <tr><td>fa</td><td><b>0.9865 ± 0.0000</b></td><td><b>0.8709 ± 0.0013</b></td><td>0.8596 ± 0.0036</td><td>0.6718 ± 0.0043</td><td>0.9228 ± 0.0281</td><td>0.8011 ± 0.0236</td></tr>
    <tr><td>fc_a</td><td>0.8413 ± 0.0017</td><td><b>0.7033 ± 0.0017</b></td><td><b>0.8422 ± 0.0022</b></td><td>0.5932 ± 0.0039</td><td>0.8130 ± 0.0120</td><td>0.6727 ± 0.0146</td></tr>
    <tr><td>hdBPMN-icdar2021</td><td><b>0.8357 ± 0.0023</b></td><td><b>0.4429 ± 0.0033</b></td><td>0.5254 ± 0.0025</td><td>0.1271 ± 0.0056</td><td>0.5648 ± 0.0273</td><td>0.3497 ± 0.0192</td></tr>
    <tr><td>sap-sam-bpmn</td><td>0.4034 ± 0.0062</td><td>0.2213 ± 0.0054</td><td><b>0.6593 ± 0.0018</b></td><td>0.2239 ± 0.0022</td><td>0.5949 ± 0.0581</td><td><b>0.3699 ± 0.0536</b></td></tr>
    <tr><td>sap-sam-uml</td><td><b>0.9572 ± 0.0043</b></td><td><b>0.2085 ± 0.0051</b></td><td>0.7763 ± 0.0172</td><td>0.0803 ± 0.0050</td><td>0.7404 ± 0.0361</td><td>0.1221 ± 0.0239</td></tr>
    <tr><td>sems</td><td><b>0.7482 ± 0.0029</b></td><td><b>0.7283 ± 0.0048</b></td><td>0.6616 ± 0.0027</td><td>0.3391 ± 0.0107</td><td>0.6667 ± 0.0189</td><td>0.6746 ± 0.0158</td></tr>
    <tr><td>synth_sem</td><td><b>0.9050 ± 0.0079</b></td><td><b>0.3938 ± 0.0041</b></td><td>0.7666 ± 0.0027</td><td>0.1721 ± 0.0074</td><td>0.4600 ± 0.0307</td><td>0.2405 ± 0.0195</td></tr>
    <tr><td>synth_fc</td><td><b>0.8058 ± 0.0031</b></td><td><b>0.7186 ± 0.0018</b></td><td>0.4146 ± 0.2315</td><td>0.3698 ± 0.2067</td><td>0.4540 ± 0.0463</td><td>0.3934 ± 0.0293</td></tr>
  </tbody>
</table>

### Isolated Edge Recognition (five-shot, Levenshtein threshold = 0)

**F1 ± std.** Node Recognition combines text and class; Edge Recognition combines endpoints, label, and class.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Gemma4</th>
      <th colspan="2">InternVL</th>
      <th colspan="2">Qwen</th>
    </tr>
    <tr>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>cbd</td><td>0.9973 ± 0.0007</td><td><b>0.9455 ± 0.0004</b></td><td><b>0.9985 ± 0.0000</b></td><td>0.8654 ± 0.0029</td><td>0.9382 ± 0.0139</td><td>0.8809 ± 0.0133</td></tr>
    <tr><td>fa</td><td><b>1.0000 ± 0.0000</b></td><td><b>0.8789 ± 0.0007</b></td><td><b>1.0000 ± 0.0000</b></td><td>0.7226 ± 0.0034</td><td>0.9314 ± 0.0184</td><td>0.8283 ± 0.0164</td></tr>
    <tr><td>fc_a</td><td><b>1.0000 ± 0.0000</b></td><td><b>0.9848 ± 0.0000</b></td><td><b>1.0000 ± 0.0000</b></td><td>0.8529 ± 0.0043</td><td>0.9722 ± 0.0071</td><td>0.9532 ± 0.0084</td></tr>
    <tr><td>hdBPMN-icdar2021</td><td><b>0.9935 ± 0.0000</b></td><td><b>0.7609 ± 0.0024</b></td><td>0.9757 ± 0.0037</td><td>0.2923 ± 0.0024</td><td>0.5962 ± 0.0251</td><td>0.4953 ± 0.0211</td></tr>
    <tr><td>sap-sam-bpmn</td><td><b>1.0000 ± 0.0000</b></td><td><b>0.7511 ± 0.0002</b></td><td>0.9952 ± 0.0000</td><td>0.5569 ± 0.0094</td><td>0.7372 ± 0.0234</td><td>0.5581 ± 0.0240</td></tr>
    <tr><td>sap-sam-uml</td><td><b>1.0000 ± 0.0000</b></td><td><b>0.2017 ± 0.0043</b></td><td>0.9952 ± 0.0000</td><td>0.1069 ± 0.0093</td><td>0.7373 ± 0.0777</td><td>0.1289 ± 0.0342</td></tr>
    <tr><td>sems</td><td><b>0.9992 ± 0.0000</b></td><td><b>0.8769 ± 0.0011</b></td><td>0.9877 ± 0.0037</td><td>0.5225 ± 0.0030</td><td>0.8392 ± 0.0204</td><td>0.7417 ± 0.0202</td></tr>
    <tr><td>synth_sem</td><td><b>0.9670 ± 0.0000</b></td><td><b>0.4622 ± 0.0045</b></td><td><b>0.9670 ± 0.0000</b></td><td>0.2790 ± 0.0055</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>synth_fc</td><td><b>1.0000 ± 0.0000</b></td><td><b>0.7600 ± 0.0008</b></td><td><b>1.0000 ± 0.0000</b></td><td>0.7167 ± 0.0026</td><td>0.4895 ± 0.0625</td><td>0.3968 ± 0.0511</td></tr>
  </tbody>
</table>

### Bounding-Box Recognition — Gemma4 vs Arrow R-CNN (IoU threshold = 0.5)
Gemma4 is mean F1 ± std across 5 runs. Node Recognition uses bounding box and class; Edge Recognition uses bounding box, class, and endpoints. Sems is excluded, since its ground-truth labels have no arrow keypoints.
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Arrow R-CNN</th>
      <th colspan="2">Gemma4</th>
    </tr>
    <tr>
      <th>Node</th><th>Edge</th>
      <th>Node</th><th>Edge</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>fa</td><td><b>1.0000</b></td><td><b>0.9728</b></td><td>0.9477 ± 0.0028</td><td>0.6210 ± 0.0124</td></tr>
    <tr><td>fc_a</td><td><b>0.9976</b></td><td><b>0.9244</b></td><td>0.8562 ± 0.0034</td><td>0.2091 ± 0.0041</td></tr>
    <tr><td>hdBPMN-icdar2021</td><td><b>0.9022</b></td><td><b>0.8337</b></td><td>0.5126 ± 0.0043</td><td>0.0257 ± 0.0014</td></tr>
    <tr><td>synth_sem</td><td>&ndash;</td><td>&ndash;</td><td>0.8610 ± 0.0013</td><td>0.2215 ± 0.0042</td></tr>
    <tr><td>synth_fc</td><td>&ndash;</td><td>&ndash;</td><td>0.7427 ± 0.0040</td><td>0.0392 ± 0.0038</td></tr>
  </tbody>
</table>
