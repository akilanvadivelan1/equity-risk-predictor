# =========================================================
# ERPSA - Loughran-McDonald Financial Sentiment Dictionary
# =========================================================
# Implements the finance-specific sentiment lexicon from:
# Loughran & McDonald (2011), "When Is a Liability Not a
# Liability?" Journal of Finance, 66(1), 35-65.
#
# Categories: Negative, Positive, Uncertainty, Litigious,
#             Constraining, Modal (Strong/Weak)
#
# Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# =========================================================

from enum import Enum
from typing import Dict, Set, Optional


class SentimentCategory(Enum):
    """Sentiment categories from the Loughran-McDonald dictionary."""
    NEGATIVE = "negative"
    POSITIVE = "positive"
    UNCERTAINTY = "uncertainty"
    LITIGIOUS = "litigious"
    CONSTRAINING = "constraining"
    MODAL_STRONG = "modal_strong"
    MODAL_WEAK = "modal_weak"



# =========================================================
# NEGATIVE WORDS (~2,355 in full dictionary)
# Words indicating unfavorable conditions in financial text.
# Representative subset covering the most impactful terms
# found in SEC 10-K Item 1A risk factor disclosures.
# =========================================================

NEGATIVE_WORDS: Set[str] = {
    # --- Financial Distress & Failure ---
    "abandon", "abandoned", "abandoning", "abandonment", "abandonments",
    "bankrupt", "bankruptcies", "bankruptcy",
    "closure", "closures", "closing", "closedown",
    "collapse", "collapsed", "collapses", "collapsing",
    "default", "defaulted", "defaulting", "defaults",
    "deficit", "deficits",
    "delinquencies", "delinquency", "delinquent",
    "fail", "failed", "failing", "fails", "failure", "failures",
    "insolvent", "insolvency", "insolvencies",
    "liquidate", "liquidated", "liquidating", "liquidation", "liquidations",
    "restructure", "restructured", "restructures", "restructuring", "restructurings",
    "writedown", "writedowns", "writeoff", "writeoffs",

    # --- Adverse Impact & Harm ---
    "adverse", "adversely", "adversity",
    "against",
    "damage", "damaged", "damages", "damaging",
    "danger", "dangerous", "dangerously", "dangers",
    "decline", "declined", "declines", "declining",
    "degrade", "degradation", "degraded", "degrading",
    "deteriorate", "deteriorated", "deteriorates", "deteriorating", "deterioration",
    "detrimental", "detrimentally",
    "diminish", "diminished", "diminishes", "diminishing",
    "disadvantage", "disadvantaged", "disadvantages", "disadvantageous",
    "disrupt", "disrupted", "disrupting", "disruption", "disruptions", "disruptive",
    "downgrade", "downgraded", "downgrades", "downgrading",
    "downturn", "downturns",
    "erode", "eroded", "erodes", "eroding", "erosion",
    "harm", "harmed", "harmful", "harming", "harms",
    "impair", "impaired", "impairing", "impairment", "impairments", "impairs",
    "impediment", "impediments",
    "jeopardize", "jeopardized", "jeopardizes", "jeopardizing", "jeopardy",
    "loss", "losses",
    "negative", "negatively",
    "worsen", "worsened", "worsening", "worse", "worst",

    # --- Inability & Weakness ---
    "inability", "unable", "unabled",
    "inadequate", "inadequately", "inadequacy", "inadequacies",
    "incapable",
    "ineffective", "ineffectively", "inefficiency", "inefficiencies", "inefficient",
    "insufficient", "insufficiently",
    "unfavorable", "unfavorably",
    "unsatisfactory",
    "unsuccessful", "unsuccessfully",
    "vulnerable", "vulnerabilities", "vulnerability",
    "weak", "weaken", "weakened", "weakening", "weakens", "weakness", "weaknesses",

    # --- Fraud, Misconduct & Legal Issues ---
    "breach", "breached", "breaches", "breaching",
    "complaint", "complaints",
    "concern", "concerned", "concerns", "concerning",
    "conviction", "convictions",
    "corrupt", "corrupted", "corruption",
    "crime", "crimes", "criminal", "criminally",
    "defraud", "defrauded", "defrauding",
    "fraud", "frauds", "fraudulent", "fraudulently",
    "guilt", "guilty",
    "illegal", "illegally", "illegality",
    "investigation", "investigations",
    "misappropriate", "misappropriated", "misappropriation",
    "misconduct",
    "misrepresent", "misrepresentation", "misrepresentations", "misrepresented",
    "misstate", "misstated", "misstatement", "misstatements",
    "negligence", "negligent", "negligently",
    "noncompliance", "noncompliant",
    "offense", "offenses",
    "penalty", "penalties", "penalize", "penalized",
    "restate", "restated", "restatement", "restatements", "restating",
    "sanction", "sanctioned", "sanctions",
    "violate", "violated", "violates", "violating", "violation", "violations",

    # --- Termination & Cessation ---
    "cease", "ceased", "ceases", "ceasing",
    "curtail", "curtailed", "curtailing", "curtailment", "curtailments",
    "discontinue", "discontinued", "discontinues", "discontinuing", "discontinuation",
    "divest", "divested", "divesting", "divestiture", "divestitures",
    "eliminate", "eliminated", "eliminates", "eliminating", "elimination",
    "exit", "exited", "exiting", "exits",
    "layoff", "layoffs",
    "shut", "shutdown", "shutdowns", "shutting",
    "suspend", "suspended", "suspending", "suspension", "suspensions",
    "terminate", "terminated", "terminates", "terminating", "termination", "terminations",

    # --- Threat & Risk Intensifiers ---
    "catastrophe", "catastrophes", "catastrophic", "catastrophically",
    "crisis", "crises",
    "critical", "critically",
    "destabilize", "destabilized", "destabilizing",
    "drastic", "drastically",
    "exacerbate", "exacerbated", "exacerbates", "exacerbating",
    "extreme", "extremely",
    "forfeit", "forfeited", "forfeiting", "forfeiture", "forfeitures",
    "grave", "gravely",
    "harsh", "harsher", "harshest", "harshly",
    "heighten", "heightened", "heightening",
    "impossible", "impossibility",
    "impede", "impeded", "impedes", "impeding",
    "irreparable", "irreparably",
    "irreversible", "irreversibly",
    "material", "materially",
    "obstacle", "obstacles",
    "preclude", "precluded", "precludes", "precluding",
    "problematic",
    "prohibit", "prohibited", "prohibiting", "prohibition", "prohibitions", "prohibitive",
    "severe", "severely", "severity",
    "significant", "significantly",
    "substantial", "substantially",
    "threat", "threaten", "threatened", "threatening", "threatens", "threats",
    "unforeseen",
}



# =========================================================
# POSITIVE WORDS (~354 in full dictionary)
# Words indicating favorable conditions in financial text.
# =========================================================

POSITIVE_WORDS: Set[str] = {
    "achieve", "achieved", "achievement", "achievements", "achieves", "achieving",
    "advantage", "advantaged", "advantageous", "advantageously", "advantages",
    "benefit", "benefited", "beneficial", "beneficially", "benefiting", "benefits",
    "best",
    "breakthrough", "breakthroughs",
    "creative", "creatively", "creativity",
    "efficiency", "efficiencies", "efficient", "efficiently",
    "enable", "enabled", "enables", "enabling",
    "enhance", "enhanced", "enhancement", "enhancements", "enhances", "enhancing",
    "exceed", "exceeded", "exceeding", "exceeds",
    "excellent", "excellence",
    "exceptional", "exceptionally",
    "favorable", "favorably",
    "gain", "gained", "gaining", "gains",
    "good", "goodwill",
    "great", "greater", "greatest", "greatly",
    "improve", "improved", "improvement", "improvements", "improves", "improving",
    "innovate", "innovated", "innovates", "innovating", "innovation", "innovations", "innovative",
    "leadership",
    "opportunities", "opportunity",
    "optimal", "optimistic", "optimize", "optimized", "optimizes", "optimizing",
    "outperform", "outperformed", "outperforming", "outperforms",
    "positive", "positively",
    "proactive", "proactively",
    "proficiency", "proficient", "proficiently",
    "profit", "profitability", "profitable", "profitably", "profited", "profiting", "profits",
    "progress", "progressed", "progresses", "progressing",
    "prosper", "prospered", "prospering", "prosperity", "prosperous",
    "rebound", "rebounded", "rebounding", "rebounds",
    "recover", "recovered", "recoveries", "recovering", "recovers", "recovery",
    "reward", "rewarded", "rewarding", "rewards",
    "smooth", "smoothly",
    "strength", "strengthen", "strengthened", "strengthening", "strengthens", "strengths",
    "strong", "stronger", "strongest", "strongly",
    "succeed", "succeeded", "succeeding", "succeeds", "success", "successes", "successful", "successfully",
    "superior",
    "surpass", "surpassed", "surpasses", "surpassing",
    "upturn", "upturns",
    "win", "winning", "wins", "won",
}



# =========================================================
# UNCERTAINTY WORDS (~297 in full dictionary)
# Words indicating ambiguity about future outcomes.
# =========================================================

UNCERTAINTY_WORDS: Set[str] = {
    "almost",
    "ambiguity", "ambiguities", "ambiguous",
    "anticipate", "anticipated", "anticipates", "anticipating",
    "apparent", "apparently",
    "appear", "appeared", "appears",
    "approximate", "approximated", "approximately", "approximates", "approximation",
    "assume", "assumed", "assumes", "assuming", "assumption", "assumptions",
    "believe", "believed", "believes", "believing",
    "conceivable", "conceivably",
    "conditional", "conditionally",
    "conjecture", "conjectured", "conjectures",
    "contingency", "contingencies", "contingent", "contingently",
    "could",
    "depend", "depended", "dependent", "depending", "depends",
    "doubt", "doubted", "doubtful", "doubts",
    "estimate", "estimated", "estimates", "estimating", "estimation", "estimations",
    "expect", "expected", "expecting", "expects", "expectation", "expectations",
    "expose", "exposed", "exposes", "exposing", "exposure", "exposures",
    "fluctuate", "fluctuated", "fluctuates", "fluctuating", "fluctuation", "fluctuations",
    "forecast", "forecasted", "forecasting", "forecasts",
    "implication", "implications",
    "imply", "implied", "implies", "implying",
    "imprecise", "imprecision",
    "indefinite", "indefinitely",
    "indeterminate",
    "indicate", "indicated", "indicates", "indicating", "indication", "indications",
    "inexact",
    "intend", "intended", "intending", "intends", "intention", "intentions",
    "likelihood",
    "may",
    "might",
    "nearly",
    "nonassessable",
    "occasionally",
    "pending",
    "perhaps",
    "plan", "planned", "planning", "plans",
    "possible", "possibly", "possibility", "possibilities",
    "potential", "potentially", "potentials",
    "predict", "predicted", "predicting", "prediction", "predictions", "predicts",
    "preliminary",
    "presumably", "presume", "presumed", "presumes", "presuming", "presumption",
    "probabilistic", "probability", "probable", "probably",
    "project", "projected", "projecting", "projection", "projections", "projects",
    "prospect", "prospects", "prospective", "prospectively",
    "random", "randomly",
    "risk", "risked", "riskier", "riskiest", "risking", "risks", "risky",
    "roughly",
    "seem", "seemed", "seemingly", "seems",
    "seldom",
    "sometimes",
    "speculate", "speculated", "speculates", "speculating", "speculation", "speculations", "speculative",
    "suggest", "suggested", "suggesting", "suggestion", "suggestions", "suggests",
    "susceptible",
    "tend", "tended", "tendency", "tendencies", "tending", "tends",
    "tentative", "tentatively",
    "uncertain", "uncertainly", "uncertainties", "uncertainty",
    "unclear",
    "undecided",
    "undefined",
    "undesignated",
    "undetermined",
    "unexpected", "unexpectedly",
    "unforeseen",
    "unknown", "unknowns",
    "unlikely",
    "unplanned",
    "unpredictable", "unpredictability",
    "unproven",
    "unquantifiable",
    "unresolved",
    "unsettled",
    "unspecified",
    "unusual", "unusually",
    "variable", "variability", "variables", "variation", "variations", "vary", "varied", "varies", "varying",
    "volatile", "volatility",
}



# =========================================================
# LITIGIOUS WORDS (~903 in full dictionary)
# Words associated with legal proceedings and litigation.
# =========================================================

LITIGIOUS_WORDS: Set[str] = {
    "accuse", "accused", "accuses", "accusing",
    "adjudicate", "adjudicated", "adjudicates", "adjudicating", "adjudication",
    "allegation", "allegations", "allege", "alleged", "allegedly", "alleges", "alleging",
    "appeal", "appealed", "appealing", "appeals",
    "arbitrate", "arbitrated", "arbitrates", "arbitrating", "arbitration", "arbitrations",
    "attorney", "attorneys",
    "claim", "claimant", "claimants", "claimed", "claiming", "claims",
    "class action",
    "complaint", "complaints",
    "compliance",
    "consent decree",
    "convicted", "convicting", "conviction", "convictions",
    "counsel",
    "counterclaim", "counterclaimed", "counterclaims",
    "court", "courts",
    "damages",
    "decree", "decreed", "decrees",
    "defendant", "defendants",
    "deposition", "depositions",
    "discovery",
    "dismiss", "dismissal", "dismissals", "dismissed", "dismisses", "dismissing",
    "dispute", "disputed", "disputes", "disputing",
    "enforce", "enforceability", "enforceable", "enforced", "enforcement", "enforces", "enforcing",
    "enjoin", "enjoined", "enjoining",
    "guilty",
    "hearing", "hearings",
    "indict", "indicted", "indicting", "indictment", "indictments",
    "infringe", "infringed", "infringement", "infringements", "infringes", "infringing",
    "injunction", "injunctions", "injunctive",
    "jury", "juries",
    "law", "laws", "lawsuit", "lawsuits",
    "legal", "legally",
    "litigant", "litigants", "litigate", "litigated", "litigates", "litigating", "litigation", "litigations",
    "plaintiff", "plaintiffs",
    "plead", "pleaded", "pleading", "pleadings", "pleads",
    "precedent", "precedents",
    "prosecute", "prosecuted", "prosecutes", "prosecuting", "prosecution", "prosecutions", "prosecutor",
    "regulation", "regulations", "regulatory",
    "remedy", "remedial", "remediate", "remediated", "remediation", "remedies",
    "restitution",
    "rule", "ruled", "rules", "ruling", "rulings",
    "settle", "settled", "settlement", "settlements", "settles", "settling",
    "statute", "statutes", "statutory",
    "subpoena", "subpoenaed", "subpoenas",
    "sue", "sued", "sues", "suing",
    "summon", "summoned", "summons",
    "testify", "testified", "testifies", "testifying", "testimonial", "testimony",
    "tribunal", "tribunals",
    "verdict", "verdicts",
    "witness", "witnesses",
}



# =========================================================
# CONSTRAINING WORDS (~184 in full dictionary)
# Words indicating limitations on actions or resources.
# =========================================================

CONSTRAINING_WORDS: Set[str] = {
    "abide", "abided", "abides", "abiding",
    "bind", "binding", "binds", "bound",
    "commit", "commitment", "commitments", "commits", "committed", "committing",
    "compel", "compelled", "compelling", "compels",
    "comply", "compliance", "complied", "complies", "complying",
    "condition", "conditional", "conditionally", "conditioned", "conditions",
    "confine", "confined", "confines", "confining",
    "constrain", "constrained", "constraining", "constrains", "constraint", "constraints",
    "curtail", "curtailed", "curtailing", "curtailment", "curtails",
    "decree", "decreed", "decrees",
    "demand", "demanded", "demanding", "demands",
    "dependent",
    "duress",
    "encumber", "encumbered", "encumbering", "encumbers", "encumbrance", "encumbrances",
    "enforce", "enforced", "enforces", "enforcing",
    "forbid", "forbidden", "forbidding", "forbids",
    "force", "forced", "forces", "forcing",
    "hamper", "hampered", "hampering", "hampers",
    "hinder", "hindered", "hindering", "hinders", "hindrance",
    "impede", "impeded", "impedes", "impeding", "impediment", "impediments",
    "impose", "imposed", "imposes", "imposing", "imposition",
    "inhibit", "inhibited", "inhibiting", "inhibits", "inhibition",
    "limit", "limitation", "limitations", "limited", "limiting", "limits",
    "mandate", "mandated", "mandates", "mandating", "mandatory",
    "must",
    "necessitate", "necessitated", "necessitates", "necessitating",
    "obligate", "obligated", "obligates", "obligating", "obligation", "obligations", "obligatory",
    "preclude", "precluded", "precludes", "precluding", "preclusion",
    "prevent", "prevented", "preventing", "prevents", "prevention",
    "prohibit", "prohibited", "prohibiting", "prohibition", "prohibitions", "prohibitive", "prohibits",
    "require", "required", "requirement", "requirements", "requires", "requiring",
    "restrict", "restricted", "restricting", "restriction", "restrictions", "restrictive", "restricts",
    "shall",
    "stipulate", "stipulated", "stipulates", "stipulating", "stipulation", "stipulations",
    "tighten", "tightened", "tightening", "tightens",
}



# =========================================================
# MODAL WORDS (Strong & Weak)
# Strong = certainty/commitment; Weak = hedging/possibility
# =========================================================

MODAL_STRONG_WORDS: Set[str] = {
    "always", "best", "clearly", "definitely", "definitively",
    "highest", "must", "never", "shall", "strongest",
    "undoubtedly", "will",
}

MODAL_WEAK_WORDS: Set[str] = {
    "almost", "apparently", "approximately", "conceivably",
    "could", "depend", "depends", "generally",
    "largely", "likely", "mainly", "may", "maybe",
    "might", "mostly", "nearly", "occasionally",
    "often", "ought", "partially", "partly",
    "perhaps", "possible", "possibly", "potentially",
    "presumably", "probably", "roughly",
    "seldom", "should", "sometimes", "somewhat",
    "suggest", "suggests", "tend", "tends",
    "typically", "unlikely", "usually",
}



# =========================================================
# LoughranMcDonaldDictionary Class — Public API
# =========================================================

class LoughranMcDonaldDictionary:
    """
    Finance-specific sentiment dictionary for analyzing SEC filings.

    Provides lookup and classification of words according to the
    Loughran-McDonald (2011) sentiment categories.

    Usage:
        lm = LoughranMcDonaldDictionary()
        categories = lm.classify_word("adversely")
        # -> {SentimentCategory.NEGATIVE}

        scores = lm.score_text("The company may face adverse conditions.")
        # -> {'negative': 1, 'uncertainty': 1, ...}
    """

    def __init__(self):
        """Initialize with all word lists."""
        self._categories: Dict[SentimentCategory, Set[str]] = {
            SentimentCategory.NEGATIVE: NEGATIVE_WORDS,
            SentimentCategory.POSITIVE: POSITIVE_WORDS,
            SentimentCategory.UNCERTAINTY: UNCERTAINTY_WORDS,
            SentimentCategory.LITIGIOUS: LITIGIOUS_WORDS,
            SentimentCategory.CONSTRAINING: CONSTRAINING_WORDS,
            SentimentCategory.MODAL_STRONG: MODAL_STRONG_WORDS,
            SentimentCategory.MODAL_WEAK: MODAL_WEAK_WORDS,
        }

        # Build reverse lookup: word -> set of categories
        self._word_to_categories: Dict[str, Set[SentimentCategory]] = {}
        for category, words in self._categories.items():
            for word in words:
                if word not in self._word_to_categories:
                    self._word_to_categories[word] = set()
                self._word_to_categories[word].add(category)

    def classify_word(self, word: str) -> Set[SentimentCategory]:
        """
        Classify a single word into its sentiment categories.

        Args:
            word: A single word (case-insensitive).

        Returns:
            Set of SentimentCategory values, or empty set if not in dictionary.
        """
        return self._word_to_categories.get(word.lower().strip(), set())


    def is_in_category(self, word: str, category: SentimentCategory) -> bool:
        """Check if a word belongs to a specific category."""
        return word.lower().strip() in self._categories.get(category, set())

    def get_category_words(self, category: SentimentCategory) -> Set[str]:
        """Get all words in a given category."""
        return self._categories.get(category, set()).copy()

    def get_category_size(self, category: SentimentCategory) -> int:
        """Get the number of words in a category."""
        return len(self._categories.get(category, set()))

    def score_text(self, text: str) -> Dict[str, int]:
        """
        Count sentiment words in a text across all categories.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary mapping category names to word counts.
            Also includes 'word_count' (total words) for density calculation.
        """
        import re
        words = re.findall(r'[a-z]+', text.lower())
        total_words = len(words)

        counts = {cat.value: 0 for cat in SentimentCategory}
        counts['word_count'] = total_words

        for word in words:
            categories = self._word_to_categories.get(word, set())
            for cat in categories:
                counts[cat.value] += 1

        return counts

    def score_text_density(self, text: str) -> Dict[str, float]:
        """
        Compute sentiment density (proportion) for each category.

        Returns:
            Dictionary mapping category names to density values (0.0 to 1.0).
            Density = category_count / total_word_count.
        """
        counts = self.score_text(text)
        total = counts.get('word_count', 1)
        if total == 0:
            total = 1

        densities = {}
        for cat in SentimentCategory:
            densities[cat.value] = counts[cat.value] / total

        densities['word_count'] = total
        return densities


    def score_sentences(self, sentences: list) -> Dict[str, float]:
        """
        Score a list of sentences and return aggregate densities.

        This is the primary interface for ERPSA Phase 2 —
        it scores the changed/new sentences from Step 2D output.

        Args:
            sentences: List of sentence strings to analyze.

        Returns:
            Dictionary with per-category densities and total counts.
        """
        if not sentences:
            return {cat.value: 0.0 for cat in SentimentCategory}

        combined_text = ' '.join(sentences)
        return self.score_text_density(combined_text)

    def compare_sentiment(
        self,
        current_text: str,
        prior_text: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare sentiment between current and prior year text.

        Computes density for both texts and returns the difference.
        A positive delta means MORE of that category in current year.

        Args:
            current_text: Current year's risk text.
            prior_text: Prior year's risk text.

        Returns:
            Dict with keys: 'current', 'prior', 'delta' — each containing
            per-category density values.
        """
        current_density = self.score_text_density(current_text)
        prior_density = self.score_text_density(prior_text)

        delta = {}
        for cat in SentimentCategory:
            key = cat.value
            delta[key] = current_density[key] - prior_density[key]

        return {
            'current': current_density,
            'prior': prior_density,
            'delta': delta,
        }

    @property
    def total_words(self) -> int:
        """Total unique words across all categories."""
        all_words = set()
        for words in self._categories.values():
            all_words.update(words)
        return len(all_words)

    def summary(self) -> str:
        """Print a summary of the dictionary contents."""
        lines = [
            "Loughran-McDonald Financial Sentiment Dictionary",
            "=" * 50,
        ]
        for cat in SentimentCategory:
            lines.append(f"  {cat.value:<15}: {self.get_category_size(cat):>4} words")
        lines.append(f"  {'TOTAL UNIQUE':<15}: {self.total_words:>4} words")
        return '\n'.join(lines)


# Module-level convenience instance
_default_dictionary: Optional[LoughranMcDonaldDictionary] = None


def get_dictionary() -> LoughranMcDonaldDictionary:
    """Get or create the singleton dictionary instance."""
    global _default_dictionary
    if _default_dictionary is None:
        _default_dictionary = LoughranMcDonaldDictionary()
    return _default_dictionary
