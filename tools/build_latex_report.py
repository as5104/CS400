"""Assemble the requested academic structure with portable LaTeX source."""
from pathlib import Path
import re,shutil

root=Path.cwd();folder=root/'output/latex/CS400_Final_Report';folder.mkdir(parents=True,exist_ok=True)
(folder/'figures').mkdir(exist_ok=True)
for path in (root/'output/final-results/report-figures').glob('*.pdf'):
    shutil.copy2(path,folder/'figures'/path.name)
md=(root/'docs/CS400_Final_Project_Report.md').read_text(encoding='utf-8')
chunks=re.split(r'^## (.+)$',md,flags=re.M)
sections={chunks[i]:chunks[i+1].replace('<!-- pagebreak -->','').strip() for i in range(1,len(chunks),2)}
def source(prefix):return next(v for k,v in sections.items() if k.startswith(prefix))
def subparts(text):
    pieces=re.split(r'^### (.+)$',text,flags=re.M)
    return {pieces[i]:pieces[i+1].strip() for i in range(1,len(pieces),2)}
problem=subparts(source('1 '));limitations=subparts(source('12 '));use=subparts(source('11 '))

def escape(text):
    chars={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(chars.get(c,c) for c in text)
def inline(text):
    text=escape(text)
    keys=['resnet','optuna','halving','nsga','cifar','corruptions','adamw']
    text=re.sub(r'\[([1-7](?:,\s*[1-7])*)\]',lambda m:r'\cite{'+','.join(keys[int(n.strip())-1] for n in m[1].split(','))+'}',text)
    text=re.sub(r'(?<!\w)--[a-z]+(?:-[a-z]+)*',lambda m:r'\texttt{'+m[0].replace('--','-{}-')+'}',text)
    text=re.sub(r'http[^\s]+',lambda m:r'\url{'+m[0].replace(r'\_','_')+'}',text)
    return text
def markdown(text,sublevel='section'):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:out.append('');continue
        if line.startswith('### '):out.append('\\'+sublevel+'{'+escape(re.sub(r'^\d+\.\d+\s+','',line[4:]))+'}');continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
            values=[[p.strip() for p in row.strip('|').split('|')] for row in rows]
            values=[r for r in values if not all(re.fullmatch('[-: ]+',p) for p in r)]
            n=len(values[0]);column='>{\\raggedright\\arraybackslash}X'
            # Long text gets a wider second column, numeric comparison tables remain compact.
            file_table=any(r[0].startswith(('configs/','data/','output/','runs/','docs/')) for r in values[1:])
            spec=(('>{\\raggedright\\arraybackslash}p{0.42\\textwidth}' if file_table else '>{\\raggedright\\arraybackslash}p{0.25\\textwidth}')+column) if n==2 else ('l'+'r'*(n-1)) if n==6 else column*n
            env='tabular*' if n==6 else 'tabularx'
            if n==6:spec=r'@{\extracolsep{\fill}}'+spec
            out+=['\\begin{table}[H]','\\centering\\small','\\renewcommand{\\arraystretch}{1.18}',
                  '\\begin{'+env+'}{\\textwidth}{'+spec+'}',r'\toprule']
            for j,row in enumerate(values):
                cells=[r'\path{'+v+'}' if v.startswith(('configs/','data/','output/','runs/','docs/')) else inline(v) for v in row]
                if j==0:cells=[r'\textbf{'+v+'}' for v in cells]
                out.append(' & '.join(cells)+r' \\')
                if j==0:out.append(r'\midrule')
            out += [r'\bottomrule','\\end{'+env+'}']
            next_line=i
            while next_line<len(lines) and not lines[next_line].strip():next_line+=1
            if next_line<len(lines) and lines[next_line].startswith('Table '):
                caption=re.sub(r'^Table \d+\.\s*','',lines[next_line])
                out.append(r'\caption{\small '+inline(caption)+'}');i=next_line+1
            out.append(r'\end{table}');continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            name=Path(match[2]).stem
            out += [r'\begin{figure}[H]',r'\centering',r'\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{figures/'+name+'.pdf}']
            next_line=i
            while next_line<len(lines) and not lines[next_line].strip():next_line+=1
            if next_line<len(lines) and lines[next_line].startswith('Figure '):
                caption=re.sub(r'^Figure \d+\.\s*','',lines[next_line])
                out.append(r'\caption{\small '+inline(caption)+'}');i=next_line+1
            out.append(r'\end{figure}');continue
        if line.startswith('```'):
            code=[]
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            i+=1;out += [r'\begin{lstlisting}']+code+[r'\end{lstlisting}'];continue
        if line.startswith('- ') or re.match(r'^\d\. ',line):
            ordered=bool(re.match(r'^\d\. ',line));items=[line[3:] if ordered else line[2:]]
            while i<len(lines) and (re.match(r'^\d\. ',lines[i]) if ordered else lines[i].startswith('- ')):
                items.append(lines[i][3:] if ordered else lines[i][2:]);i+=1
            env='enumerate' if ordered else 'itemize';out.append('\\begin{'+env+'}')
            out += [r'\item '+inline(t) for t in items];out.append('\\end{'+env+'}');continue
        if line.startswith('Figure ') or line.startswith('Table '):out.append(r'{\small '+inline(line)+r'\par}');continue
        out.append(inline(line))
    return '\n'.join(out)

preamble=r'''\documentclass[11pt,a4paper,oneside]{report}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,tabularx,array,float}
\usepackage{setspace,enumitem,listings}
\usepackage[hidelinks,breaklinks=true]{hyperref}
\usepackage{xurl}
\usepackage{titlesec}
\setstretch{1.10}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\clubpenalty=10000
\widowpenalty=10000
\brokenpenalty=10000
\setcounter{tocdepth}{1}
\renewcommand{\bibname}{References}
\makeatletter
\renewcommand*\l@chapter[2]{\addvspace{5pt}\@dottedtocline{0}{0em}{1.5em}{\bfseries #1}{\bfseries #2}}
\makeatother
\setlist{itemsep=3pt,topsep=4pt,leftmargin=*}
\titleclass{\chapter}{straight}
\titleformat{\chapter}{\normalfont\Large\bfseries}{\thechapter}{.7em}{}
\titlespacing*{\chapter}{0pt}{22pt}{12pt}
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}
\titleformat{\section}{\large\bfseries}{\thesection}{.7em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{.7em}{}
\titlespacing*{\section}{0pt}{13pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{5pt}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,frame=none}
\hypersetup{pdftitle={Local Multiobjective Search for ResNet Configurations},pdfauthor={Ankit Sarkar}}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\centering
\vspace*{12mm}
{\LARGE\bfseries Local Multiobjective Search for ResNet Configurations\par}
\vspace{8mm}
{\large A Local Framework Combining Bayesian Search, Multi-Fidelity Training and Post-Search Robustness Evaluation\par}
\vspace{14mm}
{\large CS400 Final Year Project Report\par}
\vspace{9mm}
Submitted by\par
{\large\bfseries Ankit Sarkar\par}
Roll No. 2301033\par
B.Tech in Computer Science and Engineering\par
\vspace{12mm}
Under the guidance of\par
{\large\bfseries Dr. Parashjyoti Borah\par}
Department of Computer Science and Engineering\par
\vfill
{\large\bfseries Indian Institute of Information Technology Guwahati\par}
\vspace{5mm}
November 2026
\end{titlepage}
\pagenumbering{roman}
\hypersetup{pageanchor=true}
\chapter*{Declaration}
\phantomsection
\addcontentsline{toc}{chapter}{Declaration}
I, Ankit Sarkar, Roll No. 2301033, submit this report as an account of the CS400 project carried out under the guidance of Dr. Parashjyoti Borah, Department of Computer Science and Engineering, Indian Institute of Information Technology Guwahati.

I declare that the implementation and experimental findings described in this report are presented accurately to the best of my knowledge. Established methods, datasets and software used in the project are acknowledged through references. The report distinguishes completed work from proposed extensions and states the limitations of its findings.

\vspace{25mm}
Ankit Sarkar\par
Roll No. 2301033\par
\clearpage
\chapter*{Acknowledgments}
\phantomsection
\addcontentsline{toc}{chapter}{Acknowledgments}
I thank my project guide, Dr. Parashjyoti Borah, Department of Computer Science and Engineering, Indian Institute of Information Technology Guwahati, for supervising this CS400 project.

I acknowledge the researchers whose work on residual networks, hyperparameter optimization, multi-objective selection and corruption benchmarks provided the foundation for this project. I also acknowledge the contributors to PyTorch, torchvision, Optuna, NumPy and the other open-source tools used in the implementation.

This project was developed and evaluated on my local laptop. The work helped me connect research methods with a working system and understand the importance of careful measurement, saved evidence and honest interpretation of experimental results.
\clearpage
\chapter*{Abstract}
\phantomsection
\addcontentsline{toc}{chapter}{Abstract}
'''
abstract=source('Final project report').split('### Abstract',1)[1].strip()
out=[preamble,markdown(abstract),r'\clearpage',r'\begingroup\small\setlength{\parskip}{0pt}\singlespacing',r'\tableofcontents',r'\endgroup',r'\clearpage',r'\pagenumbering{arabic}']
def chapter(title,body):out.extend(['\\chapter{'+title+'}',body])
chapter('Introduction',markdown(problem['1.1 Motivation'])+'\n'+markdown('### Project scope\n'+problem['1.4 Scope of the evidence']))
chapter('Motivation and Objectives',markdown('### Motivation\nA practical search system must balance prediction quality with the cost of finding and training models. The laptop setting also requires reliable manual sessions and evidence that can be inspected after execution. Returning several useful configurations allows the user to choose a model according to memory and training constraints.\n\n### Objectives\n'+problem['1.2 Objectives']+'\n\n### Project contribution and prior work\n'+problem['1.3 Relationship to existing work']))

theory=r'''\section{Residual learning and classification}
A residual block adds a learned transformation to a shortcut of its input. With input $x$, residual transformation $F$ and shortcut $S$, this implementation computes
\begin{equation}
 y=\operatorname{ReLU}(F(x)+S(x)).
\end{equation}
The shortcut is the identity when dimensions match and a learned projection when they change. This follows residual-learning principles \cite{resnet}. The project's depth and width choices determine how many basic blocks and channels appear in each of its four stages.

For a labelled training example with class $y$ and predicted class probabilities $p_k$, the cross-entropy loss is $-\log p_y$. The mini-batch loss is its mean across the batch. SGD or AdamW updates model parameters using gradients of this loss. Validation accuracy is measured without gradient updates.

\section{Three objectives and Pareto dominance}
Let $c$ be a configuration and $B$ its full epoch budget: 60 for CIFAR-10 or 12 for the dashboard demonstration. Let $A_B(c)$ be final validation accuracy, $P(c)$ parameter count and $T_B(c)$ accumulated training seconds. A minimization form of the search objectives is
\begin{equation}
 f(c)=\bigl(1-A_B(c),\;P(c),\;T_B(c)\bigr).
\end{equation}
Candidate $u$ dominates $v$ if $f_j(u)\leq f_j(v)$ for every objective $j$, with a strict inequality for at least one objective. The observed Pareto set contains eligible completed candidates that no other eligible completed candidate dominates. There is no weighted sum requiring a single preference between accuracy, size and time.

\section{Intermediate training measurements}
At a promotion rung $r$, the controller replaces final accuracy with a short average and final time with a projection:
\begin{equation}
 \overline{A}_r(c)=\frac{A_{r-2}(c)+A_{r-1}(c)+A_r(c)}{3},
 \qquad \widehat{T}_B(c)=T_r(c)\frac{B}{r}.
\end{equation}
The controller compares $(1-\overline{A}_r,P,\widehat{T}_B)$ only at equal rungs. Averaging smooths validation fluctuations, while the time projection estimates the full-budget cost.

\section{Crowding distance for promotion}
Within one non-dominated front, sort candidates separately by each objective. For an interior candidate $i$ on a nonconstant objective $j$, the normalized neighbor contribution is
\begin{equation}
 d_{i,j}=\frac{f_j(i+1)-f_j(i-1)}{f_j^{\max}-f_j^{\min}},
 \qquad d_i=\sum_j d_{i,j}.
\end{equation}
Boundary points receive priority, represented by infinite distance, and constant objectives contribute zero. Larger distance favors candidates in less crowded parts of the objective space \cite{nsga}. The controller first respects front rank and uses distance only when a front overflows the remaining slots. It then applies the stated deterministic tie rule.

\section{MOTPE and staged allocation}
TPE fits a distribution $l(c)$ to parameter choices with more promising outcomes and a distribution $g(c)$ to the other observations. It prefers proposed choices with a larger ratio $l(c)/g(c)$. Multi-objective TPE uses several objectives and hypervolume-based weights for the promising observations \cite{optuna}. The implementation delegates these calculations to the pinned Optuna sampler. Only completed observations are supplied, so a partial 15-epoch score is not treated as a 60-epoch result.

A separate controller allocates more epochs through successive-halving stages \cite{halving}, using the three-objective promotion rule. Only full-budget candidates enter the final archive.

\section{Corruption metrics}
For clean accuracy $A_{\mathrm{clean}}$ and corruption accuracy $A_{k,s}$ for type $k$ and severity $s$, the reported metrics are
\begin{align}
 A_{\mathrm{corr}}&=\frac{1}{75}\sum_{k=1}^{15}\sum_{s=1}^{5} A_{k,s},\\
 E_{\mathrm{corr}}&=1-A_{\mathrm{corr}},\\
 D_{\mathrm{pp}}&=100(A_{\mathrm{clean}}-A_{\mathrm{corr}}).
\end{align}
The last expression is a percentage-point drop, not a relative percentage decrease. $E_{\mathrm{corr}}$ is unnormalized mean error. Robustness is measured after search and is not a fourth optimization objective.
'''
chapter('Theoretical and Mathematical Background',theory)
out.append(r'\clearpage')
framework=[]
for prefix,title in [('2 ','Complete system flow'),('3 ','Dataset preparation and joint search space'),('4 ','Training measurement and recovery'),('5 ','MOTPE and multiobjective promotion')]:
    framework.append(('' if prefix=='2 ' else '\\section{'+title+'}\n')+markdown(source(prefix),'subsection'))
chapter('System Framework','\n'.join(framework))
chapter('Experimental Design',markdown(source('6 ')))
results=[]
for prefix,title in [('7 ','Search outcomes'),('8 ','Accuracy and the observed Pareto set'),('9 ','Held-out evaluation and dashboard demonstration')]:
    results.append('\\section{'+title+'}\n'+markdown(source(prefix),'subsection'))
chapter('Results and Findings','\n'.join(results))
discussion='''### Interpretation of the completed searches
The four-arm comparison demonstrates how proposal generation and resource allocation operate separately and together. MOTPE with pruning tried 16 configurations and completed seven, including three survivors from later cohorts. Each full-training arm completed one later four-candidate cohort after startup. These recorded decisions confirm that the staged allocation policy ran as specified.

The shared reference explains the common 92.12% best validation accuracy. The combined method's largest observed hypervolume describes the accuracy, size and measured-time coverage of its completed set. Its benefit in this experiment is expressed through that set of trade-offs, not through a new highest-accuracy model.

### Experimental conditions and interpretation
'''+limitations['12.1 Experimental limitations']+'''

### Robustness findings
The reference model's clean accuracy of 91.47% and mean corrupted accuracy of 71.584% show why post-search evaluation is part of the framework. The severity curves and saved condition scores describe changes under the chosen benchmark. The analysis reports these measured outcomes descriptively; it does not assign causal effects to optimizer or regularization choices from the 15 distinct configurations.
'''
chapter('Discussion',markdown(discussion))
utility='''### Application demonstrated by the project
The implemented application connects a labelled dataset to trained ResNet configurations and portable model exports on one computer. Users can compare the available accuracy, size and training-time trade-offs, pause work between sessions and retain the data and evidence locally.

The completed shapes demonstration exercised this application through the dashboard: dataset upload and validation, optional settings, explicit start, local training, Pareto inspection, clean-test evaluation, export and prediction. The existing CIFAR-10 archives provide a second use: inspecting and comparing previously completed experiments without repeating training. Both uses are supported by saved run records.

'''
chapter('Contextual Application and Utility',markdown(utility+source('11 ')))
chapter('Conclusion and Future Work',markdown('### Conclusion\n'+limitations['12.2 What the project establishes']+'\n\n### Future work\n'+limitations['12.3 Further work']))
out.append(r'\appendix')
out.append(r'\titleformat{\chapter}{\normalfont\Large\bfseries}{Appendix \thechapter:}{.7em}{}')
chapter('Implementation and Verification',markdown(source('10 ')))
chapter('Reproduction and Evidence Files',markdown(source('13 ')))
refs=r'''\clearpage
\begin{thebibliography}{9}
\phantomsection
\addcontentsline{toc}{chapter}{References}
\bibitem{resnet} K. He, X. Zhang, S. Ren and J. Sun. \emph{Deep Residual Learning for Image Recognition}. CVPR, 2016. \url{https://arxiv.org/abs/1512.03385}.
\bibitem{optuna} Optuna contributors. \emph{TPESampler documentation}, version 4.5.0. \href{https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html}{Online API documentation}.
\bibitem{halving} K. Jamieson and A. Talwalkar. \emph{Non-stochastic Best Arm Identification and Hyperparameter Optimization}. AISTATS, PMLR 51, pp. 240--248, 2016. \url{https://proceedings.mlr.press/v51/jamieson16.html}.
\bibitem{nsga} K. Deb, A. Pratap, S. Agarwal and T. Meyarivan. \emph{A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II}. IEEE Transactions on Evolutionary Computation, 6(2), pp. 182--197, 2002. \url{https://doi.org/10.1109/4235.996017}.
\bibitem{cifar} A. Krizhevsky, V. Nair and G. Hinton. \emph{CIFAR-10 and CIFAR-100 datasets}. \url{https://www.cs.toronto.edu/~kriz/cifar.html}.
\bibitem{corruptions} D. Hendrycks and T. Dietterich. \emph{Benchmarking Neural Network Robustness to Common Corruptions and Perturbations}. ICLR, 2019. \url{https://arxiv.org/abs/1903.12261}. Benchmark implementation: \url{https://github.com/hendrycks/robustness}.
\bibitem{adamw} I. Loshchilov and F. Hutter. \emph{Decoupled Weight Decay Regularization}. ICLR, 2019. \url{https://arxiv.org/abs/1711.05101}.
\end{thebibliography}
\end{document}
'''
out.append(refs)
(folder/'main.tex').write_text('\n'.join(out),encoding='utf-8')
(folder/'README.txt').write_text('CS400 final project report\nAuthor: Ankit Sarkar, roll 2301033\nGuide: Dr. Parashjyoti Borah\n\nCompile main.tex with pdfLaTeX (twice), XeLaTeX, or Tectonic.\nAll required figures are in figures/. Upload this whole folder to Overleaf if preferred.\nNo external bibliography tool is required.\nThe report was built from recorded project results; the figures and data are preserved in output/final-results.\n',encoding='utf-8')
print(folder/'main.tex')
