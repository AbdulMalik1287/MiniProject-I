"""Full Mini Project report (IIIT Nagpur guidelines), appended to a filled front-matter document.

Part A (cover, declarations, certificate, acknowledgement) comes from the institute template
filled by the team. This script appends:
    Part B  abstract, contents, list of figures, list of tables, abbreviations  (roman page numbers)
    Part C  chapters 1-5, references, appendices, publications              (arabic page numbers)
Guideline format: margins 1 / 1.25 / 1.5 / 0.8 in (top/bottom/left/right); each chapter on a fresh
page, ~75 mm extra top space, chapter number and title centred, 18 pt bold, ~25 mm gaps; tables and
figures numbered per chapter, table captions above, figure captions below; references numbered in
order of first citation. Every number is read from results/.

    python -m eeg.reporting.report_figures
    python -m eeg.reporting.full_report --front <front_matter.docx> --out <report.docx>
Then open in Word and update fields (F9) to fill the contents pages.
"""
import argparse
import json

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt

from eeg.paths import DOCS, RESULTS

TNR = "Times New Roman"
SITE = {"ds004504": "Thessaloniki", "brainlat": "BrainLat", "mumtaz": "Mumtaz", "aseeg": "ASEEG",
        "warsaw": "Warsaw", "semmelweis": "Semmelweis", "ds003478": "UNM", "ds004584": "Iowa",
        "ds003490": "New Mexico", "ds002778": "San Diego"}
TASK_NAME = {"alzheimers": "AD vs healthy", "ftd": "FTD vs healthy", "alzheimers_vs_ftd": "AD vs FTD",
             "mood": "Depression", "schizophrenia": "Schizophrenia", "parkinsons": "Parkinson's"}
TASKS = [("alzheimers", "AD vs healthy"), ("ftd", "FTD vs healthy"), ("alzheimers_vs_ftd", "AD vs FTD"),
         ("mood", "Depression vs healthy"), ("schizophrenia", "Schizophrenia vs healthy")]


def load(name):
    return json.loads((RESULTS / name).read_text())


# ------------------------------------------------------------------ references (order of first citation)

REFS = {
    "jeong2004": 'J. Jeong, "EEG dynamics in patients with Alzheimer\'s disease," Clinical Neurophysiology, Vol. 115, No. 7, pp. 1490-1505, Jul. 2004.',
    "park2021": 'S. M. Park, B. Jeong, D. Y. Oh, C.-H. Choi, H. Y. Jung, J.-Y. Lee, D. Lee, and J.-S. Choi, "Identification of major psychiatric disorders from resting-state electroencephalography using a machine learning approach," Frontiers in Psychiatry, Vol. 12, Art. 707581, 2021.',
    "wagh2020": 'N. Wagh and Y. Varatharajah, "EEG-GCNN: Augmenting electroencephalogram-based neurological disease diagnosis using a domain-guided graph convolutional neural network," Proceedings of Machine Learning Research (ML4H at NeurIPS), Vol. 136, pp. 367-378, Dec. 2020.',
    "brookshire2024": 'G. Brookshire, J. Kasper, N. M. Blauch, et al., "Data leakage in deep learning studies of translational EEG," Frontiers in Neuroscience, Vol. 18, Art. 1373515, 2024.',
    "liu2026": 'Y. Liu and M. Meng, "EEG-based classification of Alzheimer\'s disease and frontotemporal dementia via a dynamic threshold graph convolutional network," Information, Vol. 17, No. 7, Art. 703, Jul. 2026.',
    "tigga2026": 'N. P. Tigga, N. Kumari, and F. Alnajjar, "Bridging data scarcity and explainability in EEG based Alzheimer\'s prediction using cGAN augmented GATv2-LSTM networks," Scientific Reports, Vol. 16, Art. 34662, Sep. 2026.',
    "miltiadous2023": 'A. Miltiadous, K. D. Tzimourta, T. Afrantou, et al., "A dataset of scalp EEG recordings of Alzheimer\'s disease, frontotemporal dementia and healthy subjects from routine EEG," Data, Vol. 8, No. 6, Art. 95, 2023.',
    "olejarczyk2017": 'E. Olejarczyk and W. Jernajczyk, "Graph-based analysis of brain connectivity in schizophrenia," PLOS ONE, Vol. 12, No. 11, e0188629, 2017.',
    "mumtaz2017": 'W. Mumtaz, L. Xia, M. A. Mohd Yasin, et al., "A wavelet-based technique to predict treatment outcome for major depressive disorder," PLOS ONE, Vol. 12, No. 2, e0171409, 2017.',
    "welch1967": 'P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra," IEEE Transactions on Audio and Electroacoustics, Vol. 15, No. 2, pp. 70-73, 1967.',
    "barachant2012": 'A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Multiclass brain-computer interface classification by Riemannian geometry," IEEE Transactions on Biomedical Engineering, Vol. 59, No. 4, pp. 920-928, Apr. 2012.',
    "donoghue2020": 'T. Donoghue, M. Haller, E. J. Peterson, et al., "Parameterizing neural power spectra into periodic and aperiodic components," Nature Neuroscience, Vol. 23, pp. 1655-1665, 2020.',
    "vinck2011": 'M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia, and C. M. A. Pennartz, "An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias," NeuroImage, Vol. 55, No. 4, pp. 1548-1565, Apr. 2011.',
    "kipf2017": 'T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," International Conference on Learning Representations (ICLR), 2017.',
    "velickovic2018": 'P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, "Graph attention networks," International Conference on Learning Representations (ICLR), 2018.',
    "brody2022": 'S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," International Conference on Learning Representations (ICLR), 2022.',
    "klepl2024": 'D. Klepl, M. Wu, and F. He, "Graph neural network-based EEG classification: A survey," IEEE Transactions on Neural Systems and Rehabilitation Engineering, Vol. 32, pp. 493-503, 2024.',
    "tang2022": 'S. Tang, J. A. Dunnmon, K. Saab, et al., "Self-supervised graph neural networks for improved electroencephalographic seizure analysis," International Conference on Learning Representations (ICLR), 2022.',
    "geng2026": 'Z. Geng, W. Guo, J. Wang, Y. Ma, and Y. Zhu, "Multi-scale spatiotemporal graph neural network using brain partitioning for major depressive disorder detection," Sensors, Vol. 26, No. 9, Art. 2868, May 2026.',
    "yan2025": 'K. Yan, X. Luo, L. Ye, et al., "Automated seizure detection in epilepsy using a novel dynamic temporal-spatial graph attention network," Scientific Reports, Vol. 15, Art. 16392, May 2025.',
    "hazra2025": 'S. Hazra and S. Ghosh, "Bridging accuracy and explainability in EEG-based graph attention network for depression detection (ExPANet)," arXiv:2511.05537, 2025.',
    "shi2021": 'Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun, "Masked label prediction: Unified message passing model for semi-supervised classification," Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI), pp. 1548-1554, Aug. 2021.',
    "jiang2024": 'W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large brain model for learning generic representations with tremendous EEG data in BCI," International Conference on Learning Representations (ICLR), 2024.',
    "johnson2007": 'W. E. Johnson, C. Li, and A. Rabinovic, "Adjusting batch effects in microarray expression data using empirical Bayes methods," Biostatistics, Vol. 8, No. 1, pp. 118-127, 2007.',
    "jaramillo2024": 'A. Jaramillo-Jimenez, D. A. Tovar-Rios, Y.-J. Mantilla-Ramos, J.-F. Ochoa-Gomez, L. Bonanni, and K. Bronnick, "ComBat models for harmonization of resting-state EEG features in multisite studies," Clinical Neurophysiology, Vol. 167, pp. 241-253, Nov. 2024.',
    "he2020": 'H. He and D. Wu, "Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach," IEEE Transactions on Biomedical Engineering, Vol. 67, No. 2, pp. 399-410, Feb. 2020.',
    "zanini2018": 'P. Zanini, M. Congedo, C. Jutten, S. Said, and Y. Berthoumieu, "Transfer learning: A Riemannian geometry framework with applications to brain-computer interfaces," IEEE Transactions on Biomedical Engineering, Vol. 65, No. 5, pp. 1107-1116, May 2018.',
    "ganin2016": 'Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, "Domain-adversarial training of neural networks," Journal of Machine Learning Research, Vol. 17, No. 59, pp. 1-35, 2016.',
    "sagawa2020": 'S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, "Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization," International Conference on Learning Representations (ICLR), 2020.',
    "gemein2020": 'L. A. W. Gemein, R. T. Schirrmeister, P. Chrabaszcz, et al., "Machine-learning-based diagnostics of EEG pathology," NeuroImage, Vol. 220, Art. 117021, Oct. 2020.',
    "delorme2023": 'A. Delorme, "EEG is better left alone," Scientific Reports, Vol. 13, Art. 2372, 2023.',
    "kessler2025": 'R. Kessler, A. Enge, and M. A. Skeide, "How EEG preprocessing shapes decoding performance," Communications Biology, Vol. 8, Art. 1039, Jul. 2025.',
    "delpup2025": 'F. Del Pup, A. Zanola, L. F. Tshimanga, A. Bertoldo, L. Finos, and M. Atzori, "The more, the better? Evaluating the role of EEG preprocessing for deep learning applications," IEEE Transactions on Neural Systems and Rehabilitation Engineering, Vol. 33, pp. 1061-1070, 2025.',
    "prado2023": 'P. Prado, V. Medel, R. Gonzalez-Gomez, et al., "The BrainLat project, a multimodal neuroimaging dataset of neurodegeneration from underrepresented backgrounds," Scientific Data, Vol. 10, Art. 889, 2023.',
    "bagherzadeh2026": 'S. Bagherzadeh, M. Norouzi, F. Farokhshad, et al., "Atieh Schizophrenia EEG, a novel high-quality dataset designed to advance biomarker and machine learning research," Scientific Data, 2026.',
    "racz_semmelweis": 'Racz et al., Resting-state EEG of patients with schizophrenia and healthy controls (Semmelweis University), Zenodo, https://doi.org/10.5281/zenodo.14808296.',
    "cavanagh_unm": 'J. F. Cavanagh, EEG: Depression rest, OpenNeuro dataset ds003478, https://openneuro.org/datasets/ds003478.',
    "obeid2016": 'I. Obeid and J. Picone, "The Temple University Hospital EEG data corpus," Frontiers in Neuroscience, Vol. 10, Art. 196, 2016.',
    "babayan2019": 'A. Babayan, M. Erbey, D. Kumral, et al., "A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults," Scientific Data, Vol. 6, Art. 180308, 2019.',
    "gramfort2013": 'A. Gramfort, M. Luessi, E. Larson, et al., "MEG and EEG data analysis with MNE-Python," Frontiers in Neuroscience, Vol. 7, Art. 267, 2013.',
    "chen2010": 'Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, "Shrinkage algorithms for MMSE covariance estimation," IEEE Transactions on Signal Processing, Vol. 58, No. 10, pp. 5016-5029, Oct. 2010.',
    "ilse2018": 'M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR Vol. 80, pp. 2127-2136, 2018.',
    "defferrard2016": 'M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural networks on graphs with fast localized spectral filtering," Advances in Neural Information Processing Systems (NeurIPS), 2016.',
    "hanley1982": 'J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," Radiology, Vol. 143, No. 1, pp. 29-36, 1982.',
    "efron1993": 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, 1st Ed., Chapman and Hall, New York, 1993.',
    "platt1999": 'J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in Advances in Large Margin Classifiers, MIT Press, pp. 61-74, 1999.',
    "cohen1988": 'J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd Ed., Lawrence Erlbaum Associates, Hillsdale, NJ, 1988.',
    "fey2019": 'M. Fey and J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.',
    "paszke2019": 'A. Paszke, S. Gross, F. Massa, et al., "PyTorch: An imperative style, high-performance deep learning library," Advances in Neural Information Processing Systems (NeurIPS), 2019.',
    "pedregosa2011": 'F. Pedregosa, G. Varoquaux, A. Gramfort, et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, Vol. 12, pp. 2825-2830, 2011.',
    "jain2019": 'S. Jain and B. C. Wallace, "Attention is not explanation," Proceedings of NAACL-HLT, pp. 3543-3556, 2019.',
}
ORDER = []


def cite(*keys):
    for k in keys:
        assert k in REFS, k
        if k not in ORDER:
            ORDER.append(k)
    return "[" + ", ".join(str(ORDER.index(k) + 1) for k in keys) + "]"


# ------------------------------------------------------------------ document helpers


class Report:
    def __init__(self, doc):
        self.doc = doc
        self.ch = 0
        self.nt = self.nf = self.ne = 0
        self.width = None

    # styles ------------------------------------------------------------
    def styles(self):
        st = self.doc.styles

        def font(s, size, bold=False, italic=False):
            s.font.name = TNR
            s.font.size = Pt(size)
            s.font.bold = bold
            s.font.italic = italic
            rpr = s.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts")
                rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), TNR)

        def get(name, base=None):
            try:
                return st[name]
            except KeyError:
                s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
                if base:
                    s.base_style = st[base]
                return s

        body = get("Report Body", "Normal")
        font(body, 12)
        pf = body.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for name, size, bold, align, before, after in (
                ("Chapter Number", 18, True, WD_ALIGN_PARAGRAPH.CENTER, Pt(0), Mm(25)),
                ("Heading 1", 18, True, WD_ALIGN_PARAGRAPH.CENTER, Pt(0), Mm(25)),
                ("Report H2", 14, True, WD_ALIGN_PARAGRAPH.LEFT, Pt(12), Pt(6)),
                ("Report H3", 12, True, WD_ALIGN_PARAGRAPH.LEFT, Pt(10), Pt(4)),
                ("Front Heading", 16, True, WD_ALIGN_PARAGRAPH.CENTER, Pt(0), Pt(18)),
                ("Table Caption", 11, True, WD_ALIGN_PARAGRAPH.CENTER, Pt(10), Pt(4)),
                ("Figure Caption", 11, True, WD_ALIGN_PARAGRAPH.CENTER, Pt(2), Pt(12)),
                ("Table Note", 9.5, False, WD_ALIGN_PARAGRAPH.LEFT, Pt(2), Pt(10)),
                ("Equation", 12, False, WD_ALIGN_PARAGRAPH.LEFT, Pt(6), Pt(6)),
                ("Reference", 11, False, WD_ALIGN_PARAGRAPH.JUSTIFY, Pt(0), Pt(6)),
                ("Code", 9.5, False, WD_ALIGN_PARAGRAPH.LEFT, Pt(0), Pt(0))):
            s = get(name, "Normal")
            font(s, size, bold, italic=(name == "Table Note"))
            if name == "Code":
                s.font.name = "Consolas"
                s.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:ascii"), "Consolas")
                s.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:hAnsi"), "Consolas")
            p = s.paragraph_format
            p.alignment = align
            p.space_before = before
            p.space_after = after
            p.line_spacing = 1.0 if name in ("Table Note", "Code", "Reference") else 1.15
            p.keep_with_next = name in ("Chapter Number", "Heading 1", "Report H2", "Report H3", "Table Caption")
            p.left_indent = Pt(0)
            p.first_line_indent = Pt(0)
            s.font.color.rgb = None
        for name, lvl in (("Report H2", 1), ("Report H3", 2)):  # outline levels for Word's navigation pane
            ppr = st[name].element.get_or_add_pPr()
            ol = OxmlElement("w:outlineLvl")
            ol.set(qn("w:val"), str(lvl))
            ppr.append(ol)

    # sections ----------------------------------------------------------
    def section(self, fmt):
        s = self.doc.add_section(WD_SECTION.NEW_PAGE)
        s.top_margin, s.bottom_margin = Inches(1), Inches(1.25)
        s.left_margin, s.right_margin = Inches(1.5), Inches(0.8)
        self.width = s.page_width - s.left_margin - s.right_margin
        for part in (s.header, s.footer):
            part.is_linked_to_previous = False
            for p in part.paragraphs:
                for r in p.runs:
                    r.text = ""
        fp = s.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.field(fp, "PAGE", "1")
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        pg.set(qn("w:start"), "1")
        sp = s._sectPr
        cols = sp.find(qn("w:cols"))
        (cols.addprevious(pg) if cols is not None else sp.append(pg))
        return s

    @staticmethod
    def field(par, instr, placeholder=""):
        def fc(kind):
            r = par.add_run()
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
            r._r.append(e)
        fc("begin")
        r = par.add_run()
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = f" {instr} "
        r._r.append(it)
        if placeholder is not None:
            fc("separate")
            par.add_run(placeholder)
        fc("end")

    def tc(self, par, text, level=1):
        def fc(kind):
            r = par.add_run()
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
            r._r.append(e)
        fc("begin")
        r = par.add_run()
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = f' TC "{text}" \\l {level} '
        r._r.append(it)
        fc("end")

    def page_break(self):
        self.doc.add_paragraph(style="Report Body").add_run().add_break(WD_BREAK.PAGE)

    def spacer(self, height):
        """Fixed-height empty paragraph: Word drops 'space before' after a manual page break."""
        par = self.doc.add_paragraph(style="Report Body")
        par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        par.paragraph_format.line_spacing = height
        par.paragraph_format.space_after = Pt(0)

    # content -----------------------------------------------------------
    def front_heading(self, text, toc=True, new_page=True):
        if new_page:
            self.page_break()
        p = self.doc.add_paragraph(text, style="Front Heading")
        if toc:
            self.tc(p, text)
        return p

    def chapter(self, title):
        self.ch += 1
        self.nt = self.nf = self.ne = 0
        if self.ch > 1:
            self.page_break()
        self.spacer(Mm(70))  # ~75 mm extra top margin (guidelines 4.1)
        self.doc.add_paragraph(f"Chapter {self.ch}", style="Chapter Number")
        p = self.doc.add_paragraph(title, style="Heading 1")
        self.tc(p, f"{self.ch}  {title}")

    def unnumbered_chapter(self, title):
        self.page_break()
        self.spacer(Mm(20))
        p = self.doc.add_paragraph(title, style="Heading 1")
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Mm(12)
        self.tc(p, title)

    def h2(self, text):
        self.doc.add_paragraph(text, style="Report H2")

    def h3(self, text):
        self.doc.add_paragraph(text, style="Report H3")

    def p(self, text):
        return self.doc.add_paragraph(text, style="Report Body")

    def bullets(self, items, numbered=False):
        for i, t in enumerate(items, 1):
            par = self.doc.add_paragraph(style="Report Body")
            pf = par.paragraph_format
            pf.left_indent = Inches(0.4)
            pf.first_line_indent = Inches(-0.25)
            pf.space_after = Pt(3)
            pf.tab_stops.add_tab_stop(Inches(0.4))
            par.add_run(f"{i}." if numbered else "•")
            par.add_run("\t" + t)

    def equation(self, text):
        self.ne += 1
        par = self.doc.add_paragraph(style="Equation")
        ts = par.paragraph_format.tab_stops
        ts.add_tab_stop(int(self.width / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(int(self.width), WD_TAB_ALIGNMENT.RIGHT)
        r = par.add_run("\t" + text)
        r.italic = True
        par.add_run(f"\t({self.ch}.{self.ne})")
        return f"({self.ch}.{self.ne})"

    def table(self, caption, header, rows, widths, note=None, size=9.5):
        self.nt += 1
        self.doc.add_paragraph(f"Table {self.ch}.{self.nt}: {caption}", style="Table Caption")
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        total = sum(widths)
        for j, w in enumerate(widths):
            for cell in t.columns[j].cells:
                cell.width = int(self.width * w / total)
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                run = par.add_run(str(val))
                run.font.size = Pt(size)
                run.font.name = TNR
                run.bold = i == 0
                if i == 0:
                    sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear")
                    sh.set(qn("w:color"), "auto")
                    sh.set(qn("w:fill"), "E7ECF3")
                    cell._tc.get_or_add_tcPr().append(sh)
        for k, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            if len(rows) <= 16 and k < len(t.rows) - 1:  # short tables stay on one page
                for cell in row.cells:
                    for par in cell.paragraphs:
                        par.paragraph_format.keep_with_next = True
        tr = t.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        tr.append(hdr)
        if note:
            self.doc.add_paragraph(note, style="Table Note")
        else:
            self.doc.add_paragraph(style="Table Note")
        return f"Table {self.ch}.{self.nt}"

    def figure(self, path, caption, width=6.0):
        self.nf += 1
        par = self.doc.add_paragraph(style="Report Body")
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(path), width=Inches(min(width, self.width / 914400)))
        self.doc.add_paragraph(f"Figure {self.ch}.{self.nf}: {caption}", style="Figure Caption")
        return f"Figure {self.ch}.{self.nf}"


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def ci(r):
    return f"{r['auc']:.2f} [{r['ci'][0]:.2f}-{r['ci'][1]:.2f}]"


# ------------------------------------------------------------------ the report


def build(front, out):
    doc = Document(front)
    R = Report(doc)
    R.styles()
    fs, pdp, pv, lk = load("final_system.json"), load("pooled_deep.json"), load("prep_v2.json"), load("leakage_demo.json")
    e2, sp, tc, dt, ga = load("ensemble_v2.json"), load("site_probe.json"), load("three_class.json"), load("dtgcn.json"), load("graph_ablation.json")
    tg, cs, mc, sh = load("tigga.json"), load("cross_site.json"), load("mumtaz_check.json"), load("sharpened.json")
    ev = load("per_disorder/evaluate.json")
    ex = {t: load(f"explain_final/{t}.json") for t, _ in TASKS}
    g = lambda t, s: fs[t]["ensemble"]["per_site"][s]["auc"]  # noqa: E731
    cov = lambda t, c: fs[t]["ensemble"]["coverage"][c]["accuracy"]  # noqa: E731
    FIG = DOCS / "figures"

    # ============================== Part B ==============================
    R.section("lowerRoman")
    p = doc.add_paragraph("Abstract", style="Front Heading")
    R.tc(p, "Abstract")
    R.p("Electroencephalography (EEG) is cheap, portable and non-invasive, which makes it attractive for screening "
        "brain and mental disorders, but published EEG classifiers often report accuracies that do not survive "
        "independent testing. This project builds and rigorously evaluates graph-based and complementary models "
        "for detecting Alzheimer's disease (AD), frontotemporal dementia (FTD), depression and schizophrenia from "
        "resting-state EEG, using only public datasets and subject-wise evaluation throughout.")
    R.p("We first reproduced the EEG-GCNN baseline exactly (AUC 0.871). We then built one raw-EEG pipeline for 12 "
        "public datasets and trained about 30 models: graph convolutional and graph attention networks (GCN, "
        "GATv2, Graph Transformer, MSBP-GNN, GATv2-LSTM), Riemannian and classical classifiers, and the "
        "pretrained LaBraM transformer. The final system trains one model per disorder on all of its datasets, "
        "combines five scale-free members by mean rank, calibrates the output and may abstain when unsure.")
    R.p(f"With every subject tested by models that never saw them, the final system reaches AUC "
        f"{g('alzheimers', 'ds004504'):.2f} for AD and {g('ftd', 'ds004504'):.2f} for FTD (Thessaloniki), "
        f"{mc['auc_relative_power_1_25Hz']:.2f} for depression (Mumtaz, 1-25 Hz), "
        f"{g('schizophrenia', 'semmelweis'):.2f}-{g('schizophrenia', 'aseeg'):.2f} for schizophrenia across three "
        f"hospitals, and {g('alzheimers_vs_ftd', 'ds004504'):.2f}/{g('alzheimers_vs_ftd', 'brainlat'):.2f} for "
        f"telling AD from FTD. On its most confident half of subjects accuracy is {cov('alzheimers', '50%'):.0%} "
        f"(AD), {cov('mood', '50%'):.0%} (depression) and {cov('schizophrenia', '50%'):.0%} (schizophrenia).")
    R.p("Three findings stand out. Model complexity is not the bottleneck: logistic regression matches the graph "
        "networks, and a graph without edges performs as well as six graph constructions. Hospital fingerprints "
        "dominate: the recording site can be identified from healthy subjects alone with balanced accuracy up to "
        "1.00. Finally, window-level data splits, common in the literature, inflate every task to about 0.99 AUC, "
        "and re-running two published models showed their accuracies depend on such leakage. We also found and "
        "corrected recording confounds in two public datasets.")
    p = R.p("Keywords: EEG, graph neural networks, Alzheimer's disease, frontotemporal dementia, depression, "
            "schizophrenia, data leakage, multi-site harmonisation.")
    p.runs[0].italic = True

    R.front_heading("Table of Contents")
    R.field(doc.add_paragraph(style="Report Body"), 'TOC \\f \\h \\z \\t "Report H2,2,Report H3,3"', "Right-click and choose Update Field.")
    R.front_heading("List of Figures")
    R.field(doc.add_paragraph(style="Report Body"), 'TOC \\h \\z \\t "Figure Caption,1"', "Right-click and choose Update Field.")
    R.front_heading("List of Tables")
    R.field(doc.add_paragraph(style="Report Body"), 'TOC \\h \\z \\t "Table Caption,1"', "Right-click and choose Update Field.")

    R.front_heading("List of Abbreviations")
    R.ch = 0
    abbr = [("AD", "Alzheimer's disease"), ("AUC", "Area under the ROC curve"), ("BDI", "Beck Depression Inventory"),
            ("CAR", "Common average reference"), ("CI", "Confidence interval"), ("CV", "Cross-validation"),
            ("DANN", "Domain-adversarial neural network"), ("DT-GCN", "Dynamic threshold graph convolutional network"),
            ("EA", "Euclidean alignment"), ("ECE", "Expected calibration error"), ("EEG", "Electroencephalography"),
            ("FIR", "Finite impulse response (filter)"), ("FTD", "Frontotemporal dementia"),
            ("GAT / GATv2", "Graph attention network / its dynamic-attention variant"),
            ("GCN", "Graph convolutional network"), ("GNN", "Graph neural network"),
            ("GroupDRO", "Group distributionally robust optimisation"), ("HC", "Healthy control"),
            ("LaBraM", "Large Brain Model (pretrained EEG transformer)"), ("LOHO", "Leave one hospital out"),
            ("LR", "Logistic regression"), ("LSTM", "Long short-term memory network"),
            ("MDD", "Major depressive disorder"), ("MIL", "Multiple-instance learning"),
            ("MSBP-GNN", "Multi-scale brain-partition graph neural network"), ("OAS", "Oracle approximating shrinkage"),
            ("PLV", "Phase-locking value"), ("PSD", "Power spectral density"), ("ROC", "Receiver operating characteristic"),
            ("SZ", "Schizophrenia"), ("wPLI", "Weighted phase lag index")]
    t = doc.add_table(rows=len(abbr), cols=2)
    for i, (a, b) in enumerate(abbr):
        for j, v in enumerate((a, b)):
            c = t.cell(i, j)
            c.text = ""
            r = c.paragraphs[0].add_run(v)
            r.font.size, r.font.name, r.bold = Pt(11), TNR, j == 0
            c.width = int(R.width * (0.25 if j == 0 else 0.75))

    # ============================== Part C ==============================
    R.section("decimal")

    # ---------------- Chapter 1
    R.chapter("Introduction")
    R.h2("1.1 Background")
    R.p("Neurological and psychiatric disorders such as Alzheimer's disease, frontotemporal dementia, depression and "
        "schizophrenia affect hundreds of millions of people and are diagnosed largely through clinical interviews, "
        "cognitive testing and, for dementia, expensive imaging. Electroencephalography (EEG) records the brain's "
        "electrical activity through electrodes on the scalp. It is inexpensive, portable and non-invasive, and a "
        "few minutes of resting-state recording with the eyes closed already carries known disease markers, for "
        f"example the slowing of rhythms in Alzheimer's disease {cite('jeong2004')}. Automatic analysis of "
        "resting-state EEG could therefore support screening where specialists and imaging are scarce.")
    R.p("Machine learning on EEG has a long history, from classifiers on band-power features "
        f"{cite('park2021')} to deep networks. Graph neural networks (GNNs) are a natural fit because the EEG "
        "montage is itself a graph: electrodes are nodes, and the statistical coupling between their signals "
        f"defines edges. EEG-GCNN {cite('wagh2020')} showed that a graph convolutional network on such brain "
        "graphs can separate patients from healthy people, and many later papers report accuracies above 90%.")
    R.h2("1.2 Problem Statement")
    R.p("The problem addressed in this project is to detect four disorders - Alzheimer's disease, frontotemporal "
        "dementia, depression and schizophrenia - from short resting-state EEG recordings using graph neural "
        "networks, and to measure honestly how well such models work on people they have never seen, including "
        "people recorded at other hospitals.")
    R.h2("1.3 Motivation")
    R.p("Two observations motivated the way this work was carried out. First, public EEG datasets are small (28 "
        "to 145 people each), so complex models can easily memorise individuals. Second, a recent analysis "
        f"{cite('brookshire2024')} showed that data leakage - most often splitting the windows of one recording "
        "between training and test sets - is widespread in translational EEG and inflates reported accuracy. A "
        "useful screening model must work on a new person, so every result in this report is computed on subjects "
        "that no model saw during training or model selection.")
    R.h2("1.4 Aim and Objectives")
    R.p("The aim is a reproducible, honestly evaluated EEG system that detects each of the four disorders. The "
        "objectives were:")
    R.bullets(["reproduce a published EEG graph neural network exactly as a baseline;",
               "build a single preprocessing pipeline that turns raw recordings from any public dataset into "
               "comparable features and brain graphs;",
               "develop and compare graph models (GCN, GATv2, Graph Transformer, MSBP-GNN, GATv2-LSTM) against "
               "strong non-graph baselines and a pretrained EEG foundation model;",
               "study and, where possible, reduce the effect of differences between hospitals;",
               "re-examine published results under subject-wise evaluation;",
               "deliver a final calibrated system per disorder that can abstain when it is unsure, and explain "
               "what it relies on."], numbered=True)
    R.h2("1.5 Scope of the Work")
    R.p("The work uses only public, de-identified resting-state EEG with healthy controls recorded at the same "
        "site as the patients, reduced to the 19 electrodes of the international 10-20 system. The final scope "
        "covers four disorders on six datasets from six countries. Parkinson's disease was explored on four "
        "further datasets and then set aside to keep the project focused. The models are research software, not "
        "a diagnostic device.")
    R.h2("1.6 Contributions")
    R.bullets(["An exact reproduction of EEG-GCNN, including a fix needed to load its checkpoints in current software.",
               "A single pipeline and code base (the eeg package) covering 12 public datasets, about 30 models and "
               "every experiment in this report, with a smoke test and result files for each table.",
               "A frozen final system per disorder: five scale-free models, mean-rank ensemble, calibrated "
               "probabilities and an abstention option.",
               "Controlled evidence that, at this data scale, graph structure and model complexity add little "
               "over simple models (graph ablation with a no-edge control).",
               "Audits of two published models showing that their reported accuracies depend on test-fold "
               "selection or mixed splits, and a direct demonstration of window-level leakage on our data.",
               "Discovery and correction of recording confounds in two public datasets (BrainLat and Mumtaz)."],
              numbered=True)
    R.h2("1.7 Organization of the Report")
    R.p("Chapter 2 reviews EEG markers of the four disorders, feature-based and graph-based EEG classifiers, "
        "foundation models, multi-site harmonisation and evaluation pitfalls, and ends with the work proposed. "
        "Chapter 3 describes the datasets, preprocessing, features, graph construction, models, evaluation "
        "protocol and the final system. Chapter 4 presents and discusses the results, from the baseline "
        "reproduction to the final system and its explanation. Chapter 5 summarises the work, states the "
        "conclusions and outlines future work. Appendices list every model with its settings, the per-member "
        "results and the commands that reproduce each result.")

    # ---------------- Chapter 2
    R.chapter("Literature Review")
    R.h2("2.1 EEG Markers of Brain and Mental Disorders")
    R.p("Resting-state EEG is usually described by the power of its rhythms in frequency bands - delta (1-4 Hz), "
        "theta (4-8 Hz), alpha (8-12 Hz) and beta (12-30 Hz) - and by how strongly signals at different electrodes "
        f"are coupled. In Alzheimer's disease the spectrum shifts towards slow rhythms, with more delta and theta "
        f"and less alpha and beta, and functional connectivity decreases {cite('jeong2004')}. Frontotemporal "
        "dementia shows related but generally milder changes, which is why the two dementias are hard to tell "
        f"apart from EEG; the dataset we use for both was released with this question in mind "
        f"{cite('miltiadous2023')}. In schizophrenia, altered connectivity has been described with graph measures "
        f"{cite('olejarczyk2017')}, and in depression with spectral and wavelet features {cite('mumtaz2017')}. "
        f"Park et al. {cite('park2021')} classified six psychiatric disorders against healthy controls from "
        "band-power and coherence features of 945 people.")
    R.h2("2.2 Feature-based Machine Learning on EEG")
    R.p(f"The standard features are band powers from Welch's method {cite('welch1967')} and coherence between "
        "electrode pairs. Relative band power (each band divided by total power) cancels differences in amplifier "
        "gain between recordings. Riemannian geometry treats each recording's channel covariance matrix as a point "
        f"on a curved manifold and classifies in its tangent space {cite('barachant2012')}; it is among the most "
        "reliable EEG methods. Spectral parameterisation separates the background 1/f (aperiodic) component of "
        f"the spectrum from oscillatory peaks {cite('donoghue2020')}, and the weighted phase lag index (wPLI) "
        f"measures coupling while ignoring zero-lag volume conduction {cite('vinck2011')}.")
    R.h2("2.3 Graph Neural Networks for EEG")
    R.p(f"Graph convolutional networks (GCNs) {cite('kipf2017')} aggregate each node's neighbours with fixed "
        f"weights; graph attention networks (GAT) {cite('velickovic2018')} learn those weights, and GATv2 "
        f"{cite('brody2022')} makes the attention depend on both nodes and can read edge features. A survey of "
        f"GNN-based EEG classification {cite('klepl2024')} lists the main design choices: which electrodes are "
        "nodes, how edges are defined (distance, correlation, coherence, phase locking) and whether graphs are "
        f"fixed or learned. EEG-GCNN {cite('wagh2020')} used an 8-node bipolar graph with distance and coherence "
        f"edges and reported AUC 0.871; self-supervised graph models have been applied to seizures {cite('tang2022')}.")
    R.p("Recent disorder-specific models report high accuracy on the same public data we use. DT-GCN "
        f"{cite('liu2026')} reports 82% for three-class AD/FTD/healthy classification with a dynamic edge "
        f"threshold. Tigga et al. {cite('tigga2026')} combine GATv2 with an LSTM and cGAN augmentation for "
        f"Alzheimer's prediction. MSBP-GNN {cite('geng2026')} partitions the brain into hemisphere graphs with "
        f"multi-scale temporal convolutions for depression, and DTS-GAN {cite('yan2025')} combines graph attention "
        f"with temporal modelling for seizures. ExPANet {cite('hazra2025')} reports 97.5% for depression with "
        f"edge-gated graph attention. The Graph Transformer {cite('shi2021')} replaces attention coefficients with "
        "scaled dot-product attention over neighbours.")
    R.h2("2.4 EEG Foundation Models")
    R.p(f"LaBraM {cite('jiang2024')} is a transformer pretrained on about 2,500 hours of EEG from many sources. It "
        "splits each channel into one-second patches and learns general representations that can be fine-tuned "
        "for new tasks. Such models promise to reduce the data needed for each disorder.")
    R.h2("2.5 Multi-site Data and Harmonisation")
    R.p("EEG recorded at different hospitals differs in equipment, reference, filters and procedure. ComBat "
        f"{cite('johnson2007')}, originally for gene-expression batch effects, removes site-wise shifts and "
        f"scalings of features and has been applied to resting-state EEG {cite('jaramillo2024')}. Euclidean "
        f"alignment {cite('he2020')} whitens each subject's or site's data by its mean covariance, and Riemannian "
        f"re-centring {cite('zanini2018')} does the same on the manifold. Domain-adversarial training "
        f"{cite('ganin2016')} learns features from which a site classifier cannot recover the site, and GroupDRO "
        f"{cite('sagawa2020')} minimises the loss of the worst-performing group.")
    R.h2("2.6 Evaluation Pitfalls and Data Leakage")
    R.p(f"Brookshire et al. {cite('brookshire2024')} reviewed deep-learning studies of translational EEG and found "
        "that many split windows of the same subject between training and test sets, which lets a model "
        "recognise the person instead of the disorder and inflates accuracy. Other forms of leakage include "
        "choosing the training epoch or hyperparameters by test-set accuracy. Gemein et al. "
        f"{cite('gemein2020')} showed that with careful subject-wise evaluation, deep and feature-based models "
        "perform similarly on clinical EEG.")
    R.h2("2.7 Preprocessing Choices")
    R.p(f"Delorme {cite('delorme2023')} found that only high-pass filtering and bad-channel interpolation reliably "
        f"improved EEG analyses. Kessler et al. {cite('kessler2025')} reported that artefact correction steps "
        f"lowered decoding performance, and Del Pup et al. {cite('delpup2025')} found, on the same Alzheimer's/FTD "
        "dataset we use, that simple filtering performed as well as ICA- or ASR-based cleaning. These results "
        "suggested keeping preprocessing simple and identical across datasets.")
    R.h2("2.8 Research Gaps and Work Proposed")
    R.p("The review shows three gaps. Reported accuracies on public EEG are rarely checked under strict "
        "subject-wise evaluation; graph models are seldom compared with equally informed non-graph baselines or "
        "a no-edge control; and the effect of hospital differences is usually ignored because most studies use "
        "one dataset. This project therefore proposes to build graph models and strong baselines on one common "
        "pipeline across many public datasets, to evaluate everything subject-wise with confidence intervals, to "
        "test whether graph structure actually helps, to re-run published models under honest protocols, and to "
        "deliver a calibrated final system per disorder whose behaviour on new people is known.")

    # ---------------- Chapter 3
    R.chapter("Proposed Work")
    R.h2("3.1 Overview of the Approach")
    R.p("Figure 3.1 shows the final system. A resting-state recording from any hospital is reduced to 19 "
        "electrodes, filtered and cut into 10-second windows. Scale-free features are computed per window, five "
        "different models score the recording, and their ranks are averaged and calibrated into a probability. If "
        "the system is not confident it can answer 'unsure'. The rest of this chapter describes each step and the "
        "experiments that led to it.")
    R.figure(FIG / "report_system.png", "Final per-disorder system from recording to output.", 6.2)
    R.h2("3.2 Datasets")
    n = lambda t, s, k: fs[t]["ensemble"]["per_site"][s][k]  # noqa: E731
    R.p("All datasets are public and de-identified, and every patient group comes with healthy controls recorded at "
        "the same site - mixing patients from one hospital with controls from another would teach a model to "
        "recognise the hospital. Table 3.1 lists the datasets of the final system.")
    R.table("Datasets used in the final system", ["Dataset", "Country", "Disorder", "Patients / controls",
                                                    "Channels", "Source"],
            [["Thessaloniki (ds004504)", "Greece", "AD, FTD", f"{n('alzheimers', 'ds004504', 'n_pos')} AD, "
              f"{n('ftd', 'ds004504', 'n_pos')} FTD / {n('alzheimers', 'ds004504', 'n_neg')}", "19", cite("miltiadous2023")],
             ["BrainLat", "Argentina, Chile", "AD, FTD", f"{n('alzheimers', 'brainlat', 'n_pos')} AD, "
              f"{n('ftd', 'brainlat', 'n_pos')} FTD / {n('alzheimers', 'brainlat', 'n_neg')}", "128", cite("prado2023")],
             ["Mumtaz", "Malaysia", "Depression", f"{n('mood', 'mumtaz', 'n_pos')} / {n('mood', 'mumtaz', 'n_neg')}", "19", cite("mumtaz2017")],
             ["ASEEG", "Iran", "Schizophrenia", f"{n('schizophrenia', 'aseeg', 'n_pos')} / {n('schizophrenia', 'aseeg', 'n_neg')}", "19", cite("bagherzadeh2026")],
             ["Warsaw (RepOD)", "Poland", "Schizophrenia", f"{n('schizophrenia', 'warsaw', 'n_pos')} / {n('schizophrenia', 'warsaw', 'n_neg')}", "19", cite("olejarczyk2017")],
             ["Semmelweis", "Hungary", "Schizophrenia", f"{n('schizophrenia', 'semmelweis', 'n_pos')} / {n('schizophrenia', 'semmelweis', 'n_neg')}", "64", cite("racz_semmelweis")]],
            [1.6, 1.0, 1.0, 1.5, 0.7, 0.7],
            note="All recordings are eyes-closed resting state. BrainLat: 14 Chilean controls have no signal file on "
                 "the distribution platform and were excluded.")
    R.p(f"Further data were used in earlier phases: the EEG-GCNN corpus (TUH abnormal EEG {cite('obeid2016')} and "
        f"MPI LEMON healthy subjects {cite('babayan2019')}), the Park et al. psychiatric corpus of 945 people "
        f"{cite('park2021')} (precomputed features only), four Parkinson's datasets, and the UNM depression dataset "
        f"{cite('cavanagh_unm')}. UNM was dropped because its 'depressed' group was defined by a questionnaire score: "
        "of 46 such students only 11 had depression confirmed by clinical interview.")
    R.h2("3.3 Preprocessing Pipeline")
    R.p(f"Every recording passes through the same steps, implemented with MNE-Python {cite('gramfort2013')}:")
    R.bullets(["selection of the 19 electrodes of the 10-20 system (BrainLat's 128-channel caps are mapped to the "
               "nearest 10-20 positions);",
               "a notch filter at the local mains frequency (50 or 60 Hz) at the native sampling rate;",
               "resampling to 128 Hz and a 1-45 Hz FIR band-pass filter;",
               "re-referencing to the common average;",
               "cutting into non-overlapping 10-second windows and rejecting any window in which a channel exceeds "
               "300 microvolts peak-to-peak or is flat (standard deviation below 0.2 microvolts)."])
    R.p("No ICA or automatic artefact removal is applied, following the preprocessing evidence in Section 2.7. "
        "The first version kept the first 30 clean windows per subject. The final version (preprocessing v2) "
        "skips the first 60 seconds when at least 30 windows remain afterwards, keeps up to 60 clean windows spread "
        "evenly over the recording, and summarises each subject by the median over windows.")
    R.h2("3.4 Feature Extraction")
    R.p("For each window and channel c the power spectral density P(c, f) is estimated with Welch's method (2 s Hann "
        "segments, 50% overlap). Six bands b are used: delta 1-4, theta 4-8, alpha 8-12, beta 12-25, high beta "
        "25-30 and gamma 30-45 Hz. The log relative band power is")
    R.equation("rp(c, b) = log[ Σ_{f∈b} P(c, f) / Σ_{1≤f<45 Hz} P(c, f) ]")
    R.p("Coupling between channels i and j in band b is the magnitude-squared coherence, averaged over the band's "
        "frequency bins, where S_ij is the cross-spectrum:")
    R.equation("C_ij(b) = mean_{f∈b} |S_ij(f)|² / ( S_ii(f) S_jj(f) )")
    R.p("Preprocessing v2 adds 1/f-corrected (periodic) band power. A line log10 P = o - χ log10 f is fitted to "
        "each spectrum over 2-40 Hz, excluding the alpha range 7-14 Hz, and the periodic power of a band is the "
        "mean residual above that line:")
    R.equation("pp(c, b) = mean_{f∈b} [ log10 P(c, f) - ( o_c - χ_c log10 f ) ]")
    R.p("The weighted phase lag index, tested as an alternative connectivity measure, is "
        "wPLI_ij = |Σ Im S_ij| / Σ |Im S_ij| over segments and band bins. For the Riemannian model, filter-bank "
        f"covariance matrices (five bands) are estimated per window with oracle-approximating shrinkage "
        f"{cite('chen2010')}. All features are scale-free: they do not change if a recording is multiplied by a "
        "constant, which removes differences in amplifier gain.")
    R.h2("3.5 Graph Construction")
    R.p("Each 10-second window becomes a graph with the 19 electrodes as nodes. The node features are the six "
        "relative band powers followed by the electrode's full coherence row (19 x 6 values), 120 numbers in all, "
        "so that every node knows its connectivity profile. Edges carry the six band coherences as edge features, "
        "and each node keeps its four strongest neighbours (top-4 sparsification), chosen during tuning on the Park "
        "corpus. A subject's score is the mean over its windows.")
    R.h2("3.6 Models")
    R.h3("3.6.1 EEG-GCNN baseline")
    R.p(f"The baseline reproduces EEG-GCNN {cite('wagh2020')}: 8 bipolar-channel nodes in a complete graph with "
        "self-loops (64 edges), edge weights combining electrode distance and coherence, node features of six band "
        "powers, two GCN layers (6 to 32 to 20), batch normalisation, sum pooling and a two-layer classifier "
        "(1,156 parameters).")
    R.h3("3.6.2 GATv2 per-disorder network")
    R.p("The main graph model (Figure 3.2) uses two GATv2 layers. For node i with neighbours j, GATv2 computes "
        "attention from both endpoints and the edge features e_ij:")
    R.equation("α_ij = softmax_j( a^T LeakyReLU( W [h_i || h_j || e_ij] ) )")
    R.equation("h_i' = Σ_j α_ij W h_j")
    R.p("The first layer has four heads of width 8 and the second one head of width 20, followed by batch "
        "normalisation, concatenated mean and max pooling, dropout 0.2 and a 40-10-2 classifier (9,952 "
        "parameters). It was selected from eight candidate configurations by cross-validation on the Park "
        "training split only.")
    R.figure(FIG / "gnn_selected.png", "Selected GATv2 network and the size of each layer's output.", 5.6)
    R.h3("3.6.3 Graph Transformer, MSBP-GNN and GATv2-LSTM")
    R.p(f"The Graph Transformer uses TransformerConv layers {cite('shi2021')} of the same widths (18,912 "
        f"parameters). MSBP-GNN was rebuilt from its description {cite('geng2026')}: depthwise temporal "
        "convolutions at three scales per electrode, two hemisphere graphs of 11 nodes with cosine-similarity "
        "edges, Chebyshev graph convolutions and attention fusion (36,627 parameters). The spatio-temporal "
        f"GATv2-LSTM, inspired by {cite('yan2025')}, feeds the GATv2 embedding of six consecutive windows into an "
        "LSTM (19,457 parameters).")
    R.h3("3.6.4 Attention-based multiple-instance learning and GroupDRO")
    R.p(f"Attention-MIL {cite('ilse2018')} treats a subject as a bag of window embeddings h_k and learns which "
        "windows matter:")
    R.equation("a_k = softmax_k( w^T [ tanh(V h_k) ⊙ σ(U h_k) ] ),    z = Σ_k a_k h_k")
    R.p("GroupDRO weights the loss of each hospital-and-label group by q_g, increasing q_g for groups with high "
        "loss (q_g ← q_g exp(η L_g), η = 0.01), so that training focuses on the worst group.")
    R.h3("3.6.5 Riemannian tangent-space classifier")
    R.p("For each band, a subject's covariance matrix Σ is mapped to the tangent space at the reference mean Σ_ref "
        f"of the training subjects {cite('barachant2012')}:")
    R.equation("s = upper( log( Σ_ref^{-1/2} Σ Σ_ref^{-1/2} ) )")
    R.p("and the concatenated vectors are classified by L2-regularised logistic regression (C = 0.1). In the final "
        "system each covariance is first divided by its trace, which removes the overall signal scale.")
    R.h3("3.6.6 LaBraM foundation model")
    R.p(f"The pretrained LaBraM {cite('jiang2024')} (5.82 million parameters) receives windows resampled to 200 Hz "
        "and z-scored per recording and channel. It was used frozen (embeddings plus logistic regression) and, in "
        "the final system, with its last two of twelve transformer blocks fine-tuned (965,561 trained parameters).")
    R.h3("3.6.7 Classical baselines")
    R.p(f"Logistic regression (L2, C = 0.05), support vector machines, gradient boosting, random forests and "
        f"XGBoost were trained with scikit-learn {cite('pedregosa2011')} on subject-level features, including "
        "richer markers (1/f exponent and offset, peak alpha frequency, band ratios, spectral and permutation "
        "entropy, Hjorth parameters).")
    R.h2("3.7 Handling Differences between Hospitals")
    R.p(f"Three standard methods were tested: ComBat {cite('johnson2007')} on the features, Euclidean alignment "
        f"{cite('he2020')} of the raw signals, and a domain-adversarial GATv2 {cite('ganin2016')} with a gradient "
        "reversal layer before a hospital classifier. In the final system, band power is z-scored within each site "
        f"without using labels, Riemannian covariances are re-centred per site {cite('zanini2018')}, and all "
        "features are scale-free.")
    R.h2("3.8 Evaluation Protocol")
    R.p("Every evaluation in this report follows the same rules:")
    R.bullets(["Splits are by subject: all windows of a person are either in training or in test.",
               "Nothing is selected by looking at test subjects - no epoch, threshold, hyperparameter or ensemble "
               "member.",
               "Ensembles combine members fixed before results are seen.",
               "Within-dataset results use 5-fold stratified cross-validation over subjects. The final system uses "
               "master-model folds: one stratified 5-fold split over all subjects of a disorder, stratified by "
               "site and label, so each model is trained on every hospital and tested on held-out people from "
               "every hospital. AUC is reported per site.",
               "95% confidence intervals come from 2,000 bootstrap resamples of subjects."])
    R.p(f"The main metric is the area under the ROC curve (AUC) {cite('hanley1982')}, the probability that a "
        "randomly chosen patient is ranked above a randomly chosen healthy person (0.5 = chance). Confidence "
        f"intervals use the bootstrap {cite('efron1993')}. Calibrated probabilities come from Platt scaling "
        f"{cite('platt1999')}, p = 1 / (1 + exp(-(A s + B))), fitted with cross-fitting so that each subject's "
        "probability comes from a calibrator trained on other subjects. Calibration is measured with the Brier "
        "score and the expected calibration error over ten bins:")
    R.equation("ECE = Σ_k (n_k / N) | mean(p)_k - mean(y)_k |")
    R.p("For abstention, subjects are ranked by |p - 0.5| and accuracy is reported for the most confident 100%, 90%, "
        "..., 50% of subjects. Effect sizes for explanation use Cohen's d "
        f"{cite('cohen1988')}, computed within each site and then averaged.")
    R.h2("3.9 Final System")
    R.p("The final system was fixed before it was run. Its five members are logistic regression on relative power, "
        "coherence and periodic power; the trace-normalised Riemannian classifier; GATv2; the Graph Transformer; "
        "and fine-tuned LaBraM. For subject i, each member's out-of-fold score is converted to a rank r_m(i) among "
        "the N subjects, and the ensemble score is")
    R.equation("s(i) = (1/5) Σ_m r_m(i) / N")
    R.p("which is then calibrated as described above. Depression uses the Mumtaz dataset alone; the other "
        "disorders use all of their datasets.")
    R.h2("3.10 Implementation")
    R.p(f"The code is written in Python 3.11 with PyTorch {cite('paszke2019')}, PyTorch Geometric "
        f"{cite('fey2019')}, MNE-Python, scikit-learn, pyriemann and braindecode, and runs on a laptop with an "
        "NVIDIA RTX 2060 (6 GB). It is organised as the eeg package with four parts - data (loaders, pipeline, "
        "preprocessing v2), models (graph networks, site-invariance methods, Riemannian classifier), experiments "
        "(every evaluation in this report) and reporting (report, readout page) - and is available at "
        "github.com/AbdulMalik1287/MiniProject-I. A smoke test imports every module and runs built-in self-checks.")

    # ---------------- Chapter 4
    R.chapter("Results and Discussions")
    R.h2("4.1 Baseline Reproduction")
    R.p("EEG-GCNN was reproduced with its released checkpoints on the 478 held-out subjects of its corpus (Table "
        "4.1). Every metric matches the published values. Two portability issues had to be solved: current PyTorch "
        "Geometric stores GCN weights transposed, so loading old checkpoints needs a transpose, and the positive "
        "class must be 'diseased', since the corpus is 87% diseased.")
    R.table("EEG-GCNN reproduction (mean and standard deviation over 10 checkpoints)", ["Metric", "Published", "Reproduced"],
            [["AUC", "0.871 (0.001)", "0.871 (0.001)"], ["Precision", "0.989 (0.003)", "0.989 (0.003)"],
             ["Recall", "0.677 (0.018)", "0.677 (0.017)"], ["F1", "0.804 (0.011)", "0.804 (0.011)"],
             ["Balanced accuracy", "0.810 (0.003)", "0.810 (0.003)"]], [2, 1.5, 1.5])
    R.h2("4.2 Per-disorder Graph Networks on the Park Corpus")
    R.p("On the Park corpus, one binary GATv2 per disorder was trained and evaluated once on 180 untouched test "
        "subjects (Table 4.2). Every model detects its disorder against healthy controls, but none separates its "
        "disorder from the other disorders: the models detect psychiatric illness in general, not which illness. "
        "Logistic regression on the same features reached 0.731 for schizophrenia, against 0.938 reported by Park "
        "et al.")
    pdr = ev["per_disorder"]
    R.table("Per-disorder GATv2 on the Park test split (AUC [95% CI])", ["Disorder", "vs healthy", "vs other disorders"],
            [[d, f"{v['auc']:.3f} [{v['auc_ci'][0]:.2f}-{v['auc_ci'][1]:.2f}]",
              f"{v['auc_vs_other_disorders']:.3f} [{v['auc_vs_other_ci'][0]:.2f}-{v['auc_vs_other_ci'][1]:.2f}]"]
             for d, v in pdr.items()], [2.4, 1.6, 1.6])
    R.h2("4.3 Within-dataset Results on Raw EEG")
    R.p("Table 4.3 gives the six-member ensemble (two GNNs, two logistic regressions, Riemannian, frozen LaBraM) "
        "evaluated within each dataset. Alzheimer's (0.92) and Mumtaz depression (0.97) are strong; telling AD "
        "from FTD within one hospital is near chance (0.58).")
    keys = [k for k in ("alzheimers:ds004504", "ftd:ds004504", "alzheimers_vs_ftd:ds004504", "mood:mumtaz",
                        "mood:ds003478", "schizophrenia:aseeg", "schizophrenia:warsaw", "schizophrenia:semmelweis",
                        "parkinsons:ds004584", "parkinsons:ds003490", "parkinsons:ds002778", "parkinsons:ds007526") if k in e2]
    nm = {"alzheimers:ds004504": "AD vs healthy (Thessaloniki)", "ftd:ds004504": "FTD vs healthy (Thessaloniki)",
          "alzheimers_vs_ftd:ds004504": "AD vs FTD (Thessaloniki)", "mood:mumtaz": "Depression (Mumtaz)",
          "mood:ds003478": "Depression (UNM, later dropped)", "schizophrenia:aseeg": "Schizophrenia (ASEEG)",
          "schizophrenia:warsaw": "Schizophrenia (Warsaw)", "schizophrenia:semmelweis": "Schizophrenia (Semmelweis)",
          "parkinsons:ds004584": "Parkinson's (Iowa)", "parkinsons:ds003490": "Parkinson's (New Mexico)",
          "parkinsons:ds002778": "Parkinson's (San Diego)", "parkinsons:ds007526": "Parkinson's (Tel Aviv)"}
    mem = [("gatv2", "GATv2"), ("msbp", "MSBP"), ("lr_base", "LR"), ("lr_rich", "LR rich"), ("riemann", "Riem."), ("labram", "LaBraM")]
    R.table("Within-dataset results, six-member ensemble and members (AUC)", ["Task (dataset)", "Ensemble [95% CI]"] + [b for _, b in mem],
            [[nm[k], ci(e2[k]["ensemble"])] + [f2(e2[k][m]["auc"]) for m, _ in mem] for k in keys],
            [2.2, 1.3, 0.55, 0.55, 0.55, 0.6, 0.55, 0.6], size=8.5)
    R.h2("4.4 Do Graphs Help? Graph Ablation")
    R.p("To test whether graph structure contributes, six graph constructions were compared across three GNN "
        "families on the three-class AD/FTD/healthy task, with band power as the only node feature so that "
        "connectivity could reach the model only through the edges (Figure 4.1). The 'none' graph keeps only "
        "self-loops. It performs as well as every real graph: macro-F1 stays between 0.48 and 0.55 whatever the "
        "edges are. Together with logistic regression matching the GNNs in Table 4.3, this shows that at this data "
        "scale the information is in band power, not in the graph.")
    R.figure(FIG / "report_ablation.png", "Graph ablation on AD / FTD / healthy (macro-F1, 3 repeats of 5-fold CV).", 5.8)
    R.h2("4.5 Re-examining Published Results")
    R.p(f"DT-GCN {cite('liu2026')} reports 82% three-class accuracy on the Thessaloniki data. Its released code "
        "selects, for each fold, the training epoch with the best accuracy on the held-out fold. Re-running a port "
        "of the code (Figure 4.2) gives "
        f"{dt['paper_protocol_dynamic']['accuracy_best_epoch'][0]:.2f} with that selection, "
        f"{dt['inner_val_dynamic']['accuracy'][0]:.2f} when the epoch is chosen on an inner validation split of the "
        f"training subjects, and {dt['strict_dynamic']['accuracy'][0]:.2f} with the last epoch.")
    R.figure(FIG / "report_dtgcn.png", "DT-GCN accuracy under different epoch-selection protocols.", 5.6)
    runs = tg["runs"]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    rows = []
    for prot in ("leaky", "honest"):
        for edges in (True, False):
            gg = [x for x in runs if x["protocol"] == prot and x["edges"] == edges]
            rows.append([f"{prot} split, {'with' if edges else 'without'} edges", f3(mean([x["row_bal_acc"] for x in gg])),
                         f3(mean([x["subject_bal_acc"] for x in gg])) if prot == "honest" else "-"])
    R.p(f"Tigga et al. {cite('tigga2026')} was re-run from its released code (Table 4.4). Each input row is a single "
        "time point, so the edge list connects only the first 19 rows and removing it changes nothing, and the "
        "random row split places every subject in both training and test sets. Their reported accuracy could not "
        "be reached because their EEG-to-table step is not released.")
    R.table("Re-run of the Tigga et al. model (balanced accuracy, 3 classes, chance 0.333)",
            ["Protocol", "Row balanced accuracy", "Subject balanced accuracy"], rows, [2.6, 1.6, 1.6])
    R.h2("4.6 The Effect of Window-level Splits")
    R.p("To show directly why subject-wise splits matter, the same logistic regression was evaluated with windows "
        "split at random (as in many papers) and with subjects split (Table 4.5, Figure 4.3). Random window splits "
        "give about 0.99 AUC on every task, including AD vs FTD and Semmelweis schizophrenia, where the honest score "
        "is near chance. All windows of one person share a fingerprint - skull, alpha frequency, electrode contact, "
        "recording noise - and the same label, so the leaky model recognises people rather than disease.")
    R.table("Window-level (leaky) versus subject-level (honest) splits, same model",
            ["Task (dataset)", "People", "Leaky AUC (accuracy)", "Honest AUC (accuracy)"],
            [[f"{TASK_NAME[k.split(':')[0]]} ({SITE[k.split(':')[1]]})", str(v["people"]), f"{v['leaky']['auc']:.3f} ({v['leaky']['accuracy']:.0%})",
              f"{v['honest']['auc']:.3f} ({v['honest']['accuracy']:.0%})"] for k, v in lk.items()], [2.4, 0.8, 1.5, 1.5])
    R.figure(FIG / "report_leakage.png", "Leaky and honest AUC for the same model on seven datasets.", 5.8)
    R.h2("4.7 Hospital Effects and Cross-hospital Transfer")
    R.p("A linear model on subject-level features identifies the hospital far better than the disease, even from "
        "healthy controls alone (Table 4.6). Consistently, a GATv2 trained on one schizophrenia hospital was at "
        "chance on the other (ASEEG to Warsaw 0.53, Warsaw to ASEEG 0.51), and ComBat, Euclidean alignment and "
        "domain-adversarial training did not change this. Leave-one-hospital-out evaluation was therefore not used "
        "for the final system.")
    R.table("Hospital versus disease predictability (balanced accuracy, 5-fold CV)",
            ["Disorder (datasets)", "Hospital, all", "Hospital, controls only", "Disease"],
            [[f"{TASK_NAME[t]} ({', '.join(SITE[x] for x in v['datasets'])})", f"{v['site']['bal_acc']:.2f} (chance {v['site']['chance']:.2f})",
              f"{v['site_controls']['bal_acc']:.2f}", f"{v['disease']['bal_acc']:.2f}"] for t, v in sp.items()],
            [2.3, 1.4, 1.4, 0.9])
    R.h2("4.8 BrainLat and its Recording Confound")
    R.p(f"BrainLat {cite('prado2023')} adds AD, FTD and controls from Argentina and Chile. Its patients were exported "
        "as one file type and its controls as another, with about four times lower amplitude in controls and no "
        "overlap, so any amplitude-sensitive method separates them perfectly (within-BrainLat AUC 1.00, and 0.99 "
        "with raw Riemannian covariances). With scale-free inputs, including trace-normalised covariances, AD vs "
        f"healthy drops to {pv['alzheimers:v2_riemann_tn']['per_site']['brainlat']['auc']:.2f}, in line with careful "
        "published analyses. AD vs FTD is unaffected because both patient groups share the file type. Transfer of "
        "AD vs FTD between the two cohorts is shown in Table 4.7.")
    k1, k2 = "alzheimers_vs_ftd:ds004504->brainlat:raw", "alzheimers_vs_ftd:brainlat->ds004504:raw"
    R.table("AD vs FTD trained on one cohort and tested on the other (AUC)", ["Direction", "LR", "Riemann", "LaBraM", "Ensemble"],
            [[lab, *(f3(cs[k][m]["auc"]) for m in ("lr_base", "riemann", "labram", "ensemble"))]
             for lab, k in (("Thessaloniki to BrainLat", k1), ("BrainLat to Thessaloniki", k2))], [2.2, 0.9, 0.9, 0.9, 0.9])
    R.h2("4.9 Master Models and Transformer Components")
    R.p("Following the team's architecture plan, one model per disorder was trained on all of its datasets together, "
        "and transformer-based components were compared on identical folds (Table 4.8). The Graph Transformer ties "
        "the simple models, LaBraM fine-tuning is mixed, and attention-MIL and GroupDRO are worse on every task: with "
        "about 100 subjects per task, learning from bags of windows leaves too few training examples. Weighting "
        "hospitals equally changed nothing consistently. Pooling clearly helped AD vs FTD at both sites.")
    names = [("lr_base", "LR"), ("gatv2", "GATv2"), ("gt", "Graph Tr."), ("gt_mil", "MIL"), ("gt_mil_dro", "MIL+DRO"),
             ("labram_ft", "LaBraM-FT"), ("ensemble_all6", "Ens.")]
    rows = []
    for t, tn in TASKS:
        if t == "mood":
            continue
        for s in pdp[f"{t}:lr_base"]["per_site"]:
            if s != "all":
                rows.append([f"{tn} - {SITE[s]}"] + [f2(pdp[f"{t}:{m}"]["per_site"][s]["auc"]) for m, _ in names])
    R.table("Master models per disorder (AUC per site, before preprocessing v2)", ["Task - site"] + [b for _, b in names],
            rows, [2.2, 0.55, 0.6, 0.65, 0.55, 0.7, 0.75, 0.55], size=8.5,
            note="Depression is omitted: this run pooled UNM, which was dropped afterwards.")
    R.h2("4.10 Preprocessing v2")
    R.p("Table 4.9 compares the original and the new preprocessing with logistic regression and the Riemannian "
        "classifier on the master-model folds. Spreading windows over the recording helps where recordings are long "
        "(Warsaw, Thessaloniki FTD), periodic power gives small gains in the same direction across datasets, and wPLI "
        "is worse than coherence. The first two changes were adopted; wPLI was dropped.")
    cols = [("v1_lr_base", "LR v1"), ("v2_lr_base", "LR v2"), ("v2_lr_per", "+ periodic"), ("v2_lr_wpli", "wPLI"),
            ("v1_riemann", "Riem. raw"), ("v2_riemann_tn", "Riem. scale-free")]
    rows = []
    for t, tn in TASKS:
        for s in pv[f"{t}:v1_lr_base"]["per_site"]:
            if s != "all":
                rows.append([f"{tn} - {SITE[s]}"] + [f2(pv[f"{t}:{c}"]["per_site"][s]["auc"]) if f"{t}:{c}" in pv else "-" for c, _ in cols])
    R.table("Preprocessing v1 versus v2 (AUC per site)", ["Task - site"] + [b for _, b in cols], rows,
            [2.2, 0.6, 0.6, 0.75, 0.6, 0.7, 0.9], size=8.5,
            note="Raw Riemannian on BrainLat AD/FTD vs healthy is invalid (amplitude confound, Section 4.8).")
    R.h2("4.11 Final System")
    R.p("Table 4.10 and Figure 4.4 give the final system's results. Alzheimer's (0.94) and FTD (0.89) are strong "
        "on Thessaloniki, AD vs FTD reaches 0.86 on BrainLat, depression reaches 0.97 on Mumtaz, and schizophrenia "
        "ranges from 0.67 (Semmelweis, with only two minutes of recording per person) to 0.84 (ASEEG). Compared "
        "with the earlier master-model ensemble, the final system improves at six of nine sites and is within 0.01 "
        "at the other three.")
    rows = []
    for t, tn in TASKS:
        for s, r in fs[t]["ensemble"]["per_site"].items():
            if s != "all":
                rows.append([f"{tn} - {SITE[s]}", ci(r)] + [f2(fs[t]["per_site_" + m][s]["auc"])
                                                            for m in ("lr", "riemann_tn", "gatv2", "gt", "labram_ft")])
    R.table("Final system: ensemble and member AUC per site", ["Task - site", "Ensemble [95% CI]", "LR", "Riem.", "GATv2", "Graph Tr.", "LaBraM-FT"],
            rows, [2.1, 1.35, 0.5, 0.55, 0.6, 0.7, 0.8], size=8.5,
            note="BrainLat AD/FTD vs healthy carries the recording caveat of Section 4.8. Mumtaz carries the filter "
                 "caveat of Section 4.12 (0.95 when restricted to 1-25 Hz).")
    R.figure(FIG / "report_final_auc.png", "Final system AUC per hospital with 95% bootstrap intervals.", 5.8)
    R.p("Table 4.11 and Figure 4.5 show calibration and abstention. When the system may decline the least confident "
        f"people, accuracy on the rest rises: for Alzheimer's from {cov('alzheimers', '100%'):.0%} to "
        f"{cov('alzheimers', '50%'):.0%}, for schizophrenia from {cov('schizophrenia', '100%'):.0%} to "
        f"{cov('schizophrenia', '50%'):.0%}. The calibration error for depression is high because 58 subjects are "
        "too few to fit a calibrator well, although its ranking and accuracy are good.")
    R.table("Calibration and accuracy at each coverage (all sites)", ["Task", "Brier", "ECE", "100%", "90%", "80%", "70%", "60%", "50%"],
            [[tn, f3(fs[t]["ensemble"]["brier"]), f3(fs[t]["ensemble"]["ece"])] + [f"{cov(t, c):.0%}" for c in ("100%", "90%", "80%", "70%", "60%", "50%")]
             for t, tn in TASKS], [2.0, 0.6, 0.6, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55], size=9)
    R.figure(FIG / "report_coverage.png", "Accuracy on the subjects the system answers, as it abstains on more of them.", 5.6)
    R.h2("4.12 What the System Relies On")
    R.p("Explanations combine two independent views that must agree: within-site Cohen's d of every electrode-band "
        "feature, and the coefficients of the logistic-regression member across folds (Table 4.12). For Alzheimer's "
        "(Figure 4.6) theta power rises and alpha and beta fall over the whole scalp, the classic EEG slowing, with "
        "the same direction at both hospitals. FTD shows reduced alpha and beta, Alzheimer's has more theta than "
        "FTD, and schizophrenia shows increased right-temporal delta. Attention weights were not used as "
        f"explanations {cite('jain2019')}.")
    R.table("Fold stability, agreement with effect sizes, and largest effects", ["Task", "Fold stability", "Model vs effect", "Largest effects (Cohen's d)"],
            [[tn, f2(ex[t]["fold_stability_spearman"]), f2(ex[t]["model_vs_effect_spearman"]),
              "; ".join(f"{x['feature']} ({x['cohen_d']:+.2f})" for x in ex[t]["top_effects"][:3])] for t, tn in TASKS],
            [1.6, 0.8, 0.9, 3.0], size=8.5)
    R.figure(DOCS.parent / "results" / "explain_final" / "alzheimers.png",
             "Within-site effect sizes for Alzheimer's vs healthy (red = higher in patients).", 6.2)
    R.p("For depression the largest effect was 1/f-corrected gamma power with d of about -3, implausibly large. "
        f"Mean log10 power at 44 Hz is {mc['psd_log10']['control']['44Hz']:.2f} in Mumtaz controls against "
        f"{mc['psd_log10']['mdd']['44Hz']:.2f} in patients, while 35 Hz is similar, which indicates a low-pass "
        f"filter applied to one group only. Gamma alone gives AUC {mc['auc_gamma_only']:.2f}, but relative power "
        f"computed only within 1-25 Hz still gives {mc['auc_relative_power_1_25Hz']:.2f}, so the depression result "
        "stands at about 0.95 with this caveat.")
    R.figure(DOCS.parent / "results" / "explain_final" / "schizophrenia.png",
             "Within-site effect sizes for schizophrenia vs healthy (three hospitals).", 6.2)
    R.h2("4.13 Three-class Classification")
    R.p("Distinguishing AD, FTD and healthy people at once (Table 4.13) reaches accuracy "
        f"{tc['methods']['ensemble']['accuracy'][0]:.2f} and macro-F1 {tc['methods']['ensemble']['macro_f1'][0]:.2f} "
        "with a seven-member ensemble; most errors are FTD classified as AD.")
    mm = [("gatv2", "GATv2"), ("lr_base", "LR"), ("lr_rich", "LR rich"), ("riemann", "Riemann"), ("labram", "LaBraM"),
          ("rf", "Random forest"), ("xgb", "XGBoost"), ("ensemble", "Ensemble")]
    R.table("Three-class AD / FTD / healthy on Thessaloniki (mean over 5 repeats of 5-fold CV)",
            ["Method", "Macro-F1", "Balanced acc.", "Accuracy", "Macro AUC"],
            [[b] + [f3(tc["methods"][m][k][0]) for k in ("macro_f1", "bal_acc", "accuracy", "macro_auc_ovr")] for m, b in mm],
            [1.8, 1.0, 1.1, 1.0, 1.0])
    R.h2("4.14 Discussion")
    R.p("Three lessons emerge. First, model complexity was not the bottleneck: logistic regression matched the graph "
        "networks, edges added nothing measurable, and sequence, multiple-instance and robust-optimisation variants "
        "did not help. With 28 to 130 subjects per dataset, flexible models memorise individuals and hospitals. The "
        "gains that did appear came from better inputs (window selection, periodic power, scale-free covariances) "
        f"and from combining diverse models, consistent with Gemein et al. {cite('gemein2020')}.")
    R.p("Second, the hospital fingerprint is stronger than the disease signal. This explains why transfer to unseen "
        "hospitals failed and why the master models help only when the disease signal is shared across sites, as for "
        "AD vs FTD. Third, evaluation choices dominate reported numbers. The window-level demonstration, the DT-GCN "
        "and Tigga re-runs, and the two recording confounds all show that numbers above 0.95 on these public datasets "
        "should be checked before they are trusted.")
    R.p("The main limitations are the small datasets and their wide confidence intervals (often +/-0.10), the "
        "single depression dataset, the BrainLat patient-control confound, and the unsolved problem of generalising "
        "to a hospital the model has never seen.")

    # ---------------- Chapter 5
    R.chapter("Summary and Conclusions")
    R.h2("5.1 Summary of the Work")
    R.bullets(["The EEG-GCNN baseline was reproduced exactly (AUC 0.871).",
               "Per-disorder GATv2 networks on the Park corpus detected each disorder against healthy controls but "
               "could not separate disorders from each other.",
               "One raw-EEG pipeline was built for 12 public datasets, and about 30 models were trained and compared "
               "under subject-wise evaluation.",
               "Graph ablation, re-runs of two published models and a leakage demonstration were carried out.",
               "Hospital effects were measured and three harmonisation methods tested.",
               "Recording confounds were found and corrected in BrainLat and Mumtaz.",
               "A frozen final system per disorder was built with five scale-free members, calibration and abstention, "
               "and its decisions were explained with within-site effect sizes."], numbered=True)
    R.h2("5.2 Conclusions")
    R.bullets([f"Resting-state EEG separates Alzheimer's ({g('alzheimers', 'ds004504'):.2f}), FTD "
               f"({g('ftd', 'ds004504'):.2f}) and depression (0.95) from healthy people well within a dataset, and "
               f"schizophrenia moderately ({g('schizophrenia', 'semmelweis'):.2f}-{g('schizophrenia', 'aseeg'):.2f}).",
               "Telling Alzheimer's from FTD remains hard within one hospital (0.71) but improves when two cohorts are "
               "pooled (0.86 on BrainLat).",
               "At current public data sizes, graph structure and model complexity add little over simple models; "
               "better features and model combination add more.",
               "The hospital is easier to predict than the disease, so results within one hospital do not imply "
               "transfer to another.",
               "Window-level splits and test-fold selection inflate EEG results to about 0.99 and should be ruled out "
               "before published numbers are compared.",
               "Allowing the system to abstain raises accuracy on the answered half of subjects to 84-97% for "
               "Alzheimer's, depression and schizophrenia."], numbered=True)
    R.h2("5.3 Scope for Future Work")
    R.bullets(["Add a second clinically diagnosed depression dataset (for example MODMA) to test generalisation.",
               "Collect or obtain larger multi-site cohorts, which are needed before graph and foundation models can "
               "show their advantage.",
               "Develop harmonisation methods that make models transfer to unseen hospitals, evaluated "
               "leave-one-hospital-out.",
               "Tune hyperparameters with nested cross-validation and learn ensemble weights inside training folds.",
               "Validate the abstention threshold prospectively with clinicians."], numbered=True)

    # ---------------- References
    R.unnumbered_chapter("References")
    for i, k in enumerate(ORDER, 1):
        par = doc.add_paragraph(style="Reference")
        par.paragraph_format.left_indent = Inches(0.45)
        par.paragraph_format.first_line_indent = Inches(-0.45)
        par.add_run(f"[{i}]\t{REFS[k]}")
        par.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))

    # ---------------- Appendices
    R.unnumbered_chapter("Appendix I: Models and Settings")
    R.ch, R.nt, R.nf = "I", 0, 0
    reg = [["1", "EEG-GCNN (reproduced)", "8 bipolar nodes, GCN 6-32-20, sum pool", "1,156"],
           ["2", "GATv2 per disorder", "GATv2 120-8x4-20, mean+max pool, 150 epochs", "9,952"],
           ["3", "GCN per disorder", "GCN 120-32-20", "5,004"],
           ["4", "GATv2, raw EEG", "model 2, 20 epochs on windows", "9,952"],
           ["5", "MSBP-GNN (rebuilt)", "multi-scale temporal conv, hemisphere ChebConv", "36,627"],
           ["6", "GATv2-LSTM", "GATv2 embedding of 6 windows into LSTM(40-32)", "19,457"],
           ["7-9", "ComBat / Euclidean alignment / DANN", "model 4 on harmonised inputs / adversarial head", "9,952-10,642"],
           ["10", "Graph ablation GCN / SAGE / GATv2", "band-power nodes, 6 graph types", "1,367-2,407"],
           ["11", "DT-GCN port", "PLV graphs, GCN 20-64-32-16 + GAT", "6,547"],
           ["12", "Tigga et al. GATv2-LSTM (re-run)", "their released model", "440,323"],
           ["13-17", "LR, SVM, gradient boosting, LR rich, LR both", "subject-level features", "-"],
           ["18", "Filter-bank Riemannian", "5 bands, OAS, tangent space, LR C=0.1", "-"],
           ["19", "LaBraM frozen", "block-6 embeddings + LR", "5.8 M (frozen)"],
           ["20-21", "Random forest / XGBoost", "500 trees / 300 trees depth 3", "-"],
           ["22-24", "Ensembles v1, v2, three-class", "mean rank / mean probability", "-"],
           ["26", "Graph Transformer", "TransformerConv 120-8x4-20", "18,912"],
           ["27-28", "Attention-MIL / + GroupDRO", "gated attention over windows", "20,662"],
           ["29", "LaBraM fine-tuned", "last 2 of 12 blocks + head", "965,561 trained"],
           ["32", "Scale-free Riemannian", "trace-normalised covariances, per-site re-centring", "-"],
           ["33", "Final system", "LR, model 32, GATv2, model 26, model 29; mean rank; Platt", "-"]]
    R.table("Model registry (full settings in docs/MODEL_REGISTRY.md of the repository)", ["#", "Model", "Key settings", "Parameters"],
            reg, [0.5, 2.0, 2.8, 1.0], size=8.5)
    R.p("Training settings of the graph networks: Adam, learning rate 0.001, weight decay 0.0005, batch 64, "
        "class-weighted cross-entropy, 20 epochs on raw-EEG windows; seed 42.")

    R.unnumbered_chapter("Appendix II: Secondary Results")
    R.ch, R.nt, R.nf = "II", 0, 0
    R.table("Clinically sharper secondary contrasts (six-member ensemble AUC)", ["Contrast", "Subjects", "Ensemble [95% CI]"],
            [["UNM: BDI-II >= 20 vs BDI < 7", str(sh["unm_moderate_severe"]["n"]), ci(sh["unm_moderate_severe"]["ensemble"])],
             ["Tel Aviv: moderate Parkinson's (MDS-UPDRS III >= 33) vs healthy", str(sh["telaviv_moderate_pd"]["n"]),
              ci(sh["telaviv_moderate_pd"]["ensemble"])]], [3.4, 0.8, 1.6])
    R.p("These answer easier questions than the main tasks and do not replace them. Parkinson's disease was later "
        "set aside to keep the project focused on four disorders.")

    R.unnumbered_chapter("Appendix III: Reproducing the Results")
    R.p("All commands run from the repository root (github.com/AbdulMalik1287/MiniProject-I).")
    for line in ["pip install -r requirements.txt",
                 "python tests/smoke_test.py",
                 "python -m eeg.models.train --eval-ckpt              # EEG-GCNN reproduction",
                 "python -m eeg.experiments.per_disorder evaluate      # Park corpus",
                 "python -m eeg.data.raw_eeg features --out $OUT       # raw-EEG features",
                 "python -m eeg.experiments.ensemble --out $OUT        # within-dataset table",
                 "python -m eeg.experiments.graph_ablation --out $OUT",
                 "python -m eeg.experiments.dtgcn --out $OUT",
                 "python -m eeg.experiments.leakage_demo --out $OUT",
                 "python -m eeg.experiments.pooled_deep --out $OUT     # master models",
                 "python -m eeg.data.prep_v2 extract --out $OUT        # preprocessing v2",
                 "python -m eeg.experiments.final_system --out $OUT    # final system",
                 "python -m eeg.experiments.explain_final --out $OUT",
                 "python -m eeg.reporting.report_figures",
                 "python -m eeg.reporting.full_report --front front.docx --out report.docx"]:
        doc.add_paragraph(line, style="Code")

    R.unnumbered_chapter("List of Publications")
    R.p("No publications have resulted from this work at the time of submission.")

    # make Word refresh the contents fields when the file is opened
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)
    doc.save(out)
    print(f"wrote {out}: {len(ORDER)} references")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True, help="filled front-matter docx (Part A)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.front, args.out)


if __name__ == "__main__":
    main()
