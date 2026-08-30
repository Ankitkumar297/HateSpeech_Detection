import os
import sys
import json
from typing import Dict, Any, Optional

try:
    import joblib  # type: ignore
except ImportError:
    joblib = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

try:
    from ml_pipeline.preprocessor import TextPreprocessor
except ImportError:
    from preprocessor import TextPreprocessor  # type: ignore

# Class label mapping
CLASS_MAP: Dict[int, str] = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral Language"
}

def generate_base_dataset():
    """
    Comprehensive curated dataset representing:
    Class 0: Hate Speech (targeted hatred, discrimination, dehumanization, violence against protected attributes)
    Class 1: Offensive Language (profanity, cursing, harsh insults, vulgar slang without targeted hate)
    Class 2: Neutral Language (everyday conversation, civil discourse, academic, news, positive posts)
    """
    hate_speech = [
        "Go back to your country you filthy immigrants, you do not belong here!",
        "We must eliminate all members of that religion, they are a cancer.",
        "All people of that race are subhuman criminals and should be locked up.",
        "Death to all minorities, burn down their neighborhoods now!",
        "Women are inferior creatures with lower intelligence who should stay silent.",
        "Those disabled parasites are wasting our taxpayer money, get rid of them.",
        "I despise all foreigners, our nation is for pure bloods only.",
        "Wipe out every single one of those disgusting vermin.",
        "They deserve to be beaten and hung from the trees for their heritage.",
        "No sympathy for those animals, exterminate their entire culture.",
        "All homosexuals are sick degenerates who should be put in labor camps.",
        "Send those refugees back into the sea where they belong.",
        "You dirty ethnic trash, you will never be equal to real citizens.",
        "Cleanse our streets from these subhuman invaders before it is too late.",
        "Every single one of you should be executed for your faith.",
        "They are biologically inferior and shouldn't have human rights.",
        "White supremacist power, eradicate all other races worldwide.",
        "Terrorist scum, your whole family deserves a bullet to the head.",
        "We should round up all trans people and lock them in mental institutions.",
        "Destroy their temples and kill everyone praying inside.",
        "I will hunt down every illegal immigrant and make them bleed.",
        "Your race is genetically stupid and incapable of civilization.",
        "Nobody wants your filthy kind in our neighborhood, get out or die.",
        "Hitler was completely right about what he did to those people.",
        "Shoot anyone crossing our borders on sight, kill them all.",
        "Religious zealots need to be wiped off the face of the Earth.",
        "She is just a worthless submissive female who deserves domestic violence.",
        "These indigenous natives are savages with no right to live on our land.",
        "Gas chambers should be reopened for these disgusting creatures.",
        "Hate all people from that country, they are all thieves and murderers.",
        "Kill them all and let God sort them out, pure evil race.",
        "Dirty scum of the earth, your heritage is purely vile and poisonous.",
        "Immigrants are infesting our society like rats, exterminate them.",
        "All Jews control the world and need to be purged completely.",
        "Black people are inherently violent criminals who deserve no mercy.",
        "Islam is a disease that must be cleansed with fire and blood.",
        "Castrate all gay men to protect our children from degeneracy.",
        "Lynch those traitors for speaking against our white heritage.",
        "Foreigners are stealing our jobs and poisoning our bloodline, kill them.",
        "A woman's only purpose is reproduction, disobedience should be punished with lashes.",
        "Asians are spreading viruses and destroying our economy, attack them on sight.",
        "Natives should have been wiped out completely by our ancestors.",
        "Deport and execute every single refugee crossing our borders.",
        "Non-believers are subhuman dogs that deserve brutal death.",
        "Burn all mosques and eradicate the Muslims living near us.",
        "No mercy for that degenerate race, slaughter their women and children.",
        "Keep our race pure, execute anyone who mixes blood with outsiders.",
        "They are monkeys who escaped from the jungle, send them back or shoot them.",
        "Disability is genetic weakness, euthanize them all at birth.",
        "Gypsies are born parasites, thieves and criminals who must be eliminated."
    ]

    offensive_language = [
        "Shut the fuck up you annoying idiot, nobody asked for your opinion!",
        "What a piece of shit movie, waste of damn money.",
        "Get your lazy ass off the couch and do some work bitch.",
        "This game is fucking trash and the developers are complete morons.",
        "You are so damn stupid it actually hurts my brain to listen.",
        "Fuck off, I don't give a crap about what you think.",
        "That motherfucker cut me off on the highway!",
        "Damn, that party was crazy as hell last night bro.",
        "Why is this stupid idiot shouting so loud at the referee?",
        "Holy shit that was insane, what the hell just happened?",
        "Stop bitching about the weather and just grab an umbrella.",
        "You suck at this video game, go play with dolls you noob.",
        "He is such a prick for leaving without paying his share of the bill.",
        "I hate this fucking traffic, every driver here is an asshole.",
        "Damn it, I spilled hot coffee all over my new white shirt!",
        "You're a clown and your stupid jokes are pathetic.",
        "What a dumbass move, you completely ruined the presentation.",
        "This food tastes like absolute garbage, what the fuck.",
        "Stop being such a crybaby bastard and man up already.",
        "I swear to God this wifi is slower than shit today.",
        "Look at this arrogant jerk thinking he owns the whole company.",
        "What the fuck are you looking at? Mind your own business.",
        "She called him a bastard in front of all his coworkers.",
        "Screw this nonsense, I am walking out right now.",
        "You damn fool, you dropped the crystal vase!",
        "That greasy food gave me a terrible fucking stomach ache.",
        "Quit acting like an entitled brat and clean up your mess.",
        "Damn right we won the championship last night!",
        "This code has so many bugs it's a piece of junk.",
        "Get the hell out of my way before I lose my patience.",
        "You screwed up big time, you complete knucklehead.",
        "What kind of bullcrap excuse is that?",
        "Shut your damn mouth and listen for once in your life.",
        "That referee is blind as fuck, terrible call!",
        "Stop talking shit behind people's backs, you coward.",
        "What an absolute load of crap this meeting turned out to be.",
        "You clumsy idiot, you stepped right on my new sneakers.",
        "Damn, my phone battery just died right in the middle of a call.",
        "Quit nagging me bitch, I'll fix the faucet tomorrow.",
        "This computer is a total piece of junk, keeps freezing constantly.",
        "Get over yourself man, you are not as hot shit as you think.",
        "Fuck this rainy weather, completely ruined my weekend plans.",
        "You are being an obnoxious asshole today, tone it down.",
        "That was the most boring ass lecture I have ever attended.",
        "What the hell were you thinking when you bought this monstrosity?",
        "Stop acting like a damn fool in front of our guests.",
        "You suck at driving, learn the rules of the road idiot.",
        "Holy crap, did you see the size of that spider on the wall?",
        "This airline service is absolute dogshit, lost my baggage again.",
        "Go fuck yourself and leave me alone!"
    ]

    neutral_language = [
        "Good morning everyone! Wishing you all a productive and joyful week ahead.",
        "The artificial intelligence research paper was accepted at the annual conference.",
        "I really enjoyed reading this novel, the character development was wonderful.",
        "Could you please review the attached financial report before our 3 PM meeting?",
        "The weather forecast predicts sunny skies and warm temperatures this weekend.",
        "Machine learning models require clean and balanced training data for high accuracy.",
        "We went for a beautiful hike in the national park and took many photographs.",
        "Education and healthcare should be accessible to all communities worldwide.",
        "Thank you so much for your kind assistance and quick reply.",
        "The new museum exhibition showcases classical art from the 18th century.",
        "Please remember to submit your assignment through the online portal by Friday.",
        "Sustainable green energy solutions are essential to combating global climate change.",
        "I love making homemade pizza with fresh basil and mozzarella cheese.",
        "The library provides a quiet and peaceful environment for studying and reading.",
        "Public transportation in this city is efficient, clean, and always punctual.",
        "Scientists discovered a new species of marine life in the deep Pacific Ocean.",
        "Great presentation today team, let's keep this momentum going for Q4.",
        "Volunteering at the local animal shelter has been a very rewarding experience.",
        "The orchestra delivered a mesmerizing performance of Beethoven's Symphony.",
        "Healthy nutrition and regular exercise contribute significantly to overall well-being.",
        "The software update includes bug fixes, stability improvements, and UI refinements.",
        "Can we schedule a call to discuss the collaboration opportunities next Tuesday?",
        "The sunset over the horizon painted the sky in shades of purple and gold.",
        "Learning a new language opens up fascinating cultural perspectives and friendships.",
        "The children had a wonderful time building sandcastles at the beach.",
        "Innovation in renewable solar power is driving down electricity costs.",
        "I appreciate your constructive feedback on my latest graphic design project.",
        "The university offers comprehensive scholarship programs for talented students.",
        "Farmers markets offer fresh organic produce directly from local growers.",
        "Deep neural networks and natural language processing are transforming modern technology.",
        "Hope you have a safe flight and enjoy your family vacation!",
        "The company announced a new initiative to reduce its carbon footprint by 50 percent.",
        "Let us collaborate respectfully and find a solution that benefits everyone.",
        "The garden is blooming with roses, tulips, and lavender this spring.",
        "Regular backups prevent data loss during unexpected system hardware failures.",
        "The documentary explores the historical evolution of ancient architecture.",
        "We enjoyed a delightful dinner with our close friends and family last evening.",
        "Practicing mindfulness and meditation helps reduce daily stress and anxiety.",
        "The international space station orbits Earth approximately every ninety minutes.",
        "Please feel free to ask questions at the end of the presentation.",
        "The new electric vehicle models feature enhanced battery range and smart navigation.",
        "Thank you for reaching out, we will get back to you within 24 business hours.",
        "Clean drinking water and proper sanitation are fundamental human rights.",
        "Our team successfully completed the project milestone ahead of schedule.",
        "The morning breeze was refreshing as we walked along the lake shore.",
        "Reading books daily expands vocabulary and improves analytical thinking skills.",
        "The concert tickets went on sale this morning at nine o'clock.",
        "Continuous learning and skill development are vital in today's dynamic job market.",
        "Have a wonderful weekend and take time to recharge and relax.",
        "The research team published their findings in a prestigious scientific journal."
    ]

    data = []
    for text in hate_speech:
        data.append((text, 0))
    for text in offensive_language:
        data.append((text, 1))
    for text in neutral_language:
        data.append((text, 2))

    df = pd.DataFrame(data, columns=["text", "class"])
    return df

def train_and_evaluate(dataset_csv=None, output_dir="models"):
    """
    Trains CountVectorizer + Logistic Regression and Decision Tree models,
    evaluates both, and exports model artifacts and performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = TextPreprocessor()

    if dataset_csv and os.path.exists(dataset_csv):
        print(f"Loading external dataset from {dataset_csv}...")
        df = pd.read_csv(dataset_csv)
    else:
        print("Using curated high-accuracy multi-class dataset...")
        df = generate_base_dataset()

    print(f"Dataset total samples: {len(df)}")
    print("Class distribution:")
    print(df['class'].value_counts())

    # Preprocess all texts
    print("Preprocessing text samples (lowercasing, cleaning, stopwords, SnowballStemmer)...")
    df['clean_text'] = df['text'].apply(preprocessor.tokenize_and_stem)

    # Train / Test split
    X = df['clean_text']
    y = df['class']
    
    # Vectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Logistic Regression Model
    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_report = classification_report(y_test, lr_preds, target_names=["Hate Speech", "Offensive Language", "Neutral Language"], output_dict=True, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()

    # 2. Decision Tree Classifier
    print("Training Decision Tree Model...")
    dt_model = DecisionTreeClassifier(
        max_depth=25,
        random_state=42,
        class_weight='balanced'
    )
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_report = classification_report(y_test, dt_preds, target_names=["Hate Speech", "Offensive Language", "Neutral Language"], output_dict=True, zero_division=0)
    dt_cm = confusion_matrix(y_test, dt_preds).tolist()

    # Retrain final primary classifier on all data for production deployment
    final_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1
    )
    X_all_vec = final_vectorizer.fit_transform(X)
    
    final_lr = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    final_lr.fit(X_all_vec, y)

    final_dt = DecisionTreeClassifier(
        max_depth=25,
        random_state=42,
        class_weight='balanced'
    )
    final_dt.fit(X_all_vec, y)

    # Save artifacts
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    classifier_path = os.path.join(output_dir, "classifier.joblib")
    dt_path = os.path.join(output_dir, "decision_tree.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(final_vectorizer, vectorizer_path)
    joblib.dump(final_lr, classifier_path)
    joblib.dump(final_dt, dt_path)

    metrics_data = {
        "dataset_info": {
            "total_samples": len(df),
            "classes": CLASS_MAP,
            "class_counts": {CLASS_MAP[k]: int(v) for k, v in df['class'].value_counts().to_dict().items()},
            "vocabulary_size": len(final_vectorizer.vocabulary_)
        },
        "logistic_regression": {
            "name": "Logistic Regression (Primary Classifier)",
            "accuracy": round(float(lr_acc) * 100, 2),
            "report": lr_report,
            "confusion_matrix": lr_cm,
            "hyperparameters": {
                "C": 1.5,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced"
            }
        },
        "decision_tree": {
            "name": "Decision Tree Classifier",
            "accuracy": round(float(dt_acc) * 100, 2),
            "report": dt_report,
            "confusion_matrix": dt_cm,
            "hyperparameters": {
                "max_depth": 15,
                "criterion": "gini",
                "class_weight": "balanced"
            }
        },
        "preprocessing_pipeline": [
            "Lowercase conversion",
            "URL and Mention stripping (@user, #hashtag, https://)",
            "Punctuation and special character removal",
            "NLTK English Stopwords filtering",
            "NLTK Snowball Stemming",
            "CountVectorizer N-Gram (1, 2) Tokenization"
        ]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Models successfully trained and saved to '{output_dir}' directory.")
    print(f"Logistic Regression Test Accuracy: {lr_acc*100:.2f}%")
    print(f"Decision Tree Test Accuracy: {dt_acc*100:.2f}%")
    return metrics_data

if __name__ == "__main__":
    train_and_evaluate()
