from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    """A test question and the legal units that answer it.

    `expected` holds citation prefixes: "Article 5" accepts any chunk of Article 5,
    "Article 5(1)(f)" only that point. An empty tuple means the system should refuse.
    """

    question: str
    expected: tuple[str, ...]


QUESTIONS = [
    # Definitions
    EvalQuestion("What is an AI system?", ("Article 3(1)",)),
    EvalQuestion("Who counts as a provider under the AI Act?", ("Article 3(3)",)),
    EvalQuestion("What is a deployer?", ("Article 3(4)",)),
    # Scope and timing
    EvalQuestion("Does the AI Act apply to AI used for military purposes?", ("Article 2(3)",)),
    EvalQuestion("When does the regulation apply?", ("Article 113",)),
    EvalQuestion("What does the AI Act say about AI literacy of staff?", ("Article 4",)),
    # Prohibited practices
    EvalQuestion("Which AI practices are prohibited?", ("Article 5",)),
    EvalQuestion("Is emotion recognition at work allowed?", ("Article 5(1)(f)",)),
    EvalQuestion("Is social scoring of people allowed?", ("Article 5(1)(c)",)),
    EvalQuestion(
        "Can police use real-time facial recognition in public spaces?", ("Article 5(1)(h)",)
    ),
    # High-risk systems
    EvalQuestion("When is an AI system classified as high-risk?", ("Article 6",)),
    EvalQuestion("Which areas are listed as high-risk?", ("Annex III",)),
    EvalQuestion("Is AI used to screen job applicants high-risk?", ("Annex III, point 4",)),
    EvalQuestion("What risk management do high-risk AI systems need?", ("Article 9",)),
    EvalQuestion("What are the requirements for training data?", ("Article 10",)),
    EvalQuestion(
        "What must the technical documentation contain?", ("Article 11", "Annex IV")
    ),
    EvalQuestion("Do high-risk AI systems have to keep logs?", ("Article 12",)),
    EvalQuestion("What human oversight is required for high-risk AI?", ("Article 14",)),
    EvalQuestion(
        "What are the accuracy and cybersecurity requirements?", ("Article 15",)
    ),
    EvalQuestion(
        "What obligations do deployers of high-risk AI systems have?", ("Article 26",)
    ),
    EvalQuestion("When is a fundamental rights impact assessment needed?", ("Article 27",)),
    EvalQuestion("How does conformity assessment work?", ("Article 43",)),
    EvalQuestion("What are the rules for the CE marking?", ("Article 48",)),
    EvalQuestion(
        "Where must high-risk AI systems be registered?", ("Article 49", "Article 71")
    ),
    EvalQuestion("How must serious incidents be reported?", ("Article 73",)),
    # Transparency and general-purpose AI
    EvalQuestion("Must a chatbot tell people they are talking to an AI?", ("Article 50(1)",)),
    EvalQuestion("Do deepfakes have to be labelled?", ("Article 50(4)",)),
    EvalQuestion(
        "When does a general-purpose AI model have systemic risk?", ("Article 51", "Annex XIII")
    ),
    EvalQuestion(
        "What obligations do providers of general-purpose AI models have?", ("Article 53",)
    ),
    # Innovation and enforcement
    EvalQuestion("What are AI regulatory sandboxes?", ("Article 57",)),
    EvalQuestion("What are the fines for using prohibited AI practices?", ("Article 99(3)",)),
    # Asked by reference or exact legal term, as lawyers often do
    EvalQuestion("What does Article 113 say?", ("Article 113",)),
    EvalQuestion("Summarise Annex III.", ("Annex III",)),
    EvalQuestion("What does Recital 27 say?", ("Recital (27)",)),
    EvalQuestion(
        "Which models are presumed to have high impact capabilities at 10^25 floating point "
        "operations?",
        ("Article 51",),
    ),
    EvalQuestion("What is a post-remote biometric identification system?", ("Article 3(43)",)),
    # Out of scope: the system should refuse
    EvalQuestion("What is the capital of France?", ()),
    EvalQuestion("How do I bake sourdough bread?", ()),
]
