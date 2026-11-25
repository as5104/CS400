"""Retain the approved diagram's routes, with the executed main protocol."""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

out=Path('output/final-results/report-figures/system-flow.pdf')
W,H=480,650
c=canvas.Canvas(str(out),pagesize=(W,H))
c.setTitle('Implemented eight-phase system flow')
c.setAuthor('Ankit Sarkar')
title=ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=10,leading=12)
body=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=11)
small=ParagraphStyle('small',fontName='Helvetica',fontSize=8.3,leading=10)
def para(text,x,y,w,style):
    p=Paragraph(text,style);_,h=p.wrap(w,1000);p.drawOn(c,x,H-y-h);return y+h
def node(x,y,w,h,head,text,side=False):
    c.setLineWidth(.7);c.rect(x,H-y-h,w,h)
    bottom=para(head,x+8,y+6,w-16,small if side else title)+3
    bottom=para(text,x+8,bottom,w-16,small if side else body)
    assert bottom<=y+h-3,(head,bottom,y+h)
def arrow(points,dashed=False,head=True):
    pts=[(x,H-y) for x,y in points];c.setLineWidth(.8)
    c.setDash(3,2) if dashed else c.setDash()
    p=c.beginPath();p.moveTo(*pts[0])
    for q in pts[1:]:p.lineTo(*q)
    c.drawPath(p);c.setDash()
    if head:
        x,y=pts[-1];angle=math.atan2(y-pts[-2][1],x-pts[-2][0]);size=4
        p=c.beginPath();p.moveTo(x,y)
        p.lineTo(x-size*math.cos(angle-.5),y-size*math.sin(angle-.5))
        p.lineTo(x-size*math.cos(angle+.5),y-size*math.sin(angle+.5));p.close();c.drawPath(p,fill=1,stroke=0)
def label(text,x,y,size=8):
    c.setFont('Helvetica',size);c.drawString(x,H-y,text)
def vertical(text,x,y,size=8):
    c.saveState();c.translate(x,H-y);c.rotate(90);c.setFont('Helvetica',size);c.drawString(0,0,text);c.restoreState()

X,NW,CX=48,270,183
node(X,6,NW,49,'1. Input and prepare data','Labelled images + active search budget.<br/>Validate images; save splits and preprocessing.')
node(333,6,111,67,'<b>Held-out data</b>','Clean test images;<br/>CIFAR-10-C images<br/>for evaluation only.',True)
arrow([(318,31),(333,31)],True)
arrow([(CX,55),(CX,77)]);label('Training / validation + settings',CX+7,69)
node(X,77,NW,69,'2. Joint search space + MOTPE','ResNet architecture + optimizer + regularization.<br/>First: 4 startup models trained to 60 epochs.<br/>Then: propose 4 per cohort from completed<br/>full-budget results only.')
arrow([(CX,146),(CX,168)]);label('Candidate configurations',CX+7,160)
node(X,168,NW,61,'3. Train and measure','Train on training data; evaluate on validation.<br/>Record accuracy, parameters, time + checkpoint.<br/>One candidate at a time on the local GPU.')
arrow([(CX,229),(CX,254)]);label('Cohort: equal-rung measurements',CX+7,244)
node(X,254,NW,88,'4. Multi-fidelity decision','4 at 15 epochs -&gt; 2 at 30 -&gt; 1 at 60.<br/>At 15 / 30: last-three-epoch mean validation<br/>accuracy, parameters, estimated full-budget time.<br/>Non-dominated rank -&gt; crowding -&gt; tie rule.<br/>Compare all valid candidates at the same rung.')
node(333,262,111,67,'<b>Stopped / failed</b>','Keep separate records.<br/>No MOTPE feedback<br/>or final Pareto entry.',True)
arrow([(318,294),(333,294)])
arrow([(48,299),(35,299),(35,210),(48,210)]);vertical('Promote: resume',29,290)
arrow([(48,188),(12,188),(12,392),(48,392)]);vertical('Startup: 60 epochs without pruning',7,381)
arrow([(CX,342),(CX,364)]);label('60 epochs completed',CX+7,356)
node(X,364,NW,62,'5. Completed-result archive','Weights, configuration, final validation accuracy,<br/>parameter count + actual training time.<br/>Completed history guides the next cohort.')
arrow([(318,395),(454,395),(454,111),(318,111)]);vertical('Budget remains: update MOTPE; start next cohort',463,388)
arrow([(CX,426),(CX,447),(122,447),(122,468)])
arrow([(CX,447),(354,447),(354,468)])
label('Search ends: freeze eligible results',CX+7,440)
label('Completed metrics',52,461);label('All eligible completed models',279,461)
node(22,468,206,74,'6. Observed Pareto set','Maximize final validation accuracy.<br/>Minimize parameters + actual time.<br/>Keep all non-dominated completed<br/>configurations.')
node(246,468,215,74,'7. Robustness + characterization','Clean test + 15 corruptions x 5 severities.<br/>Mean error; clean-to-corrupt drop.<br/>Descriptive analysis: 15 distinct models;<br/>no fitted predictor or SHAP findings.')
arrow([(444,40),(475,40),(475,507),(461,507)],True)
arrow([(122,542),(122,551),(240,551),(240,560)])
arrow([(354,542),(354,551),(240,551)],head=False)
node(22,560,439,55,'8. Local dashboard and model export','Pareto models + accuracy / size / time comparisons + robustness results.<br/>Export weights, configuration, class names and preprocessing; local inference.')
para('Solid: data / control flow. Dashed: held-out evaluation data only.<br/>Clean test and corruption results never guide MOTPE, promotion or Pareto selection.',22,626,439,small)
c.save();print(out)
