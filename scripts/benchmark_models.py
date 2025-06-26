#!/usr/bin/env python3
"""
Summora - Benchmark des modèles Whisper
Comparaison automatisée tiny/base/small/medium/large avec métriques WER/ROUGE
Usage: python benchmark_models.py data/audio-om-mercato-test.mp3
Usage: python benchmark_models.py data/test-reunion.mp3
"""
import argparse
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
sys.path.append('..')

# Imports Summora
from src.core.transcriber import transcribe_meeting_audio
from src.core.metrics.evaluator import create_business_evaluator
from src.core.utils import validate_audio_path, format_duration

# Ground truth datasets
GROUND_TRUTH_DATA = {
    "audio-om-mercato-test.mp3": """Alors que le mercato a ouvert ses portes dimanche à minuit, les rumeurs de transferts se font de plus en plus nombreuses au fil des jours, et ça va encore s'intensifier dans les prochaines semaines évidemment.
Aujourd'hui, on va faire un point complet sur l'actualité des derniers jours, en s'intéressant notamment aux profils de Marcus Rashford et Kevin De Bruyne, et en se demandant si ce sont des pistes crédibles.
On parlera bien évidemment de l'intérêt pour Noa Lang et de la rumeur persistante qui mène à Igor Paixão.
On évoquera aussi le cas Bennacer, pour qui la tendance serait au départ, l'intérêt de l'Inter pour Roberto De Zerbi, la clause libératoire qui lui permettrait de partir pour un montant plus que raisonnable, et le profil de Souffian El Karouani, qui plairait à la direction.

Comme d'habitude, petite question pour vous avant de commencer la vidéo :
Quel est votre avis sur la rumeur Igor Paixão ? Est-ce que vous pensez que c'est une piste crédible ? Trop chère pour nos finances ?
Et plus généralement, pensez-vous qu'il est le profil idéal pour l'OM version De Zerbi ? J'attends vos retours en commentaires.

On commence sans plus tarder avec plusieurs infos sur le centre de formation.
Tout d'abord, le compte Twitter @BelOM affirme que Yacine Badaoui a signé son premier contrat stagiaire pro avec l'OM. C'est un jeune joueur que je ne connais pas du tout, en dehors du fait qu'il jouait pour le FC Rouen, et qu'il est présenté comme un joueur très prometteur.
Il est de la génération 2008 et va désormais tenter de développer son potentiel à Marseille.

De leur côté, les médias de Tempo OM croient savoir que Félix Bienc pourrait signer son premier contrat pro à l'OM dans les prochains jours.
C'est un latéral gauche de la génération 2007, qui était notamment pisté par Monaco et Amiens, mais qui aurait finalement opté pour le projet marseillais.
On le verra certainement à l'œuvre en Youth League si son arrivée se confirme.
J'ai entendu parler de lui à quelques reprises, et a priori, c'est un joueur avec un potentiel très intéressant également. Donc à suivre.

Je termine cette partie sur le centre de formation avec un troisième joueur qui pourrait intégrer nos équipes de jeunes :
Le média stéphanois Peuple Vert révèle que Fodé Camara pourrait signer lui aussi son premier contrat pro à l'OM.
Là encore, on parle d'un jeune défenseur central de 17 ans, formé à Saint-Étienne, qui a du potentiel.
On espère qu'il s'imposera au plus haut niveau comme Saliba ou Fofana avant lui.

Au rang des départs maintenant, on a récemment parlé de l'accord de 25 millions d'euros trouvé entre l'OM et l'Inter pour le transfert de Luis Henrique.
L'Équipe apporte une précision intéressante en rappelant que l'OM devra reverser 10 % du montant à ses deux anciens clubs : 5 % pour Botafogo et 5 % également pour Três Passos.
Ça correspondrait à environ 1,25 million d'euros répartis entre les deux clubs.

Dans le même temps, on parlait dans les derniers épisodes de l'avenir de Bennacer, qui semble s'écrire en pointillés.
Il sort de 6 mois très mitigés à Marseille, malgré 2 ou 3 bons matchs à son arrivée.
Mais depuis, ses prestations ont été beaucoup plus timides — je pense qu'on sera tous d'accord là-dessus.
La faute à une forme physique loin d'être optimale. Il ne faut pas oublier qu'il avait très peu joué avec Milan sur les six premiers mois de la saison, à cause d'une blessure au mollet qui l'a éloigné des terrains d'août à décembre.

En tout cas, Nicolas Chirat croit savoir que la tendance serait plutôt à un départ, puisque les dirigeants marseillais auraient des doutes à son sujet, au point de ne pas lever l'option d'achat.
C'est un joueur que j'apprécie beaucoup, et que j'imagine capable d'apporter une vraie plus-value sportive s'il retrouve tous ses moyens après une bonne prépa.
Malgré tout, je peux comprendre les doutes de Benatia et Longoria : il a une tendance à se blesser et son salaire annuel serait de 4,2 millions d'euros net.
C'est un risque non négligeable, sachant qu'un salaire aussi élevé implique un statut de titulaire indiscutable dans le projet actuel.

Il y a pas mal de paramètres à prendre en compte :
Est-ce qu'il pourra enchaîner les matchs sans rechute ? Est-ce qu'il retrouvera son meilleur niveau et pourra le maintenir sur la durée ?
Perso, j'ai envie d'y croire — il n'a que 27 ans — mais l'absence de certitudes pourrait lui coûter sa place dans le projet.
Selon La Provence, l'OM envisage plusieurs solutions : un nouveau prêt, une renégociation de l'indemnité et du salaire, ou tout simplement ne pas lever l'option d'achat.

De son côté, Roberto De Zerbi fait partie des coachs ciblés par l'Inter en cas de départ de Simone Inzaghi, si on en croit le très fiable Gianluca Di Marzio.
Pasquale Guarro ajoute que De Zerbi serait même le profil préféré de Piero Ausilio, le directeur sportif de l'Inter.
D'après Fabrizio Romano, il y a des chances qu'Inzaghi quitte le club, étant donné que l'Arabie Saoudite le suit de près et aurait formulé une offre très intéressante sur le plan financier.

Pour autant, dans le courrier adressé aux membres du peuple bleu et blanc, on a eu une confirmation implicite que McCourt, les dirigeants, et De Zerbi partageaient la même vision du projet.
Florian Germain a d'ailleurs confirmé dans la foulée qu'il resterait la saison prochaine.
Même si certains clubs sont intéressés par son profil, j'ai la sensation qu'il veut vraiment s'inscrire dans la durée, en jouant cette Ligue des Champions qu'il a contribué à aller chercher.
Il a toujours clamé son envie de rester, mais aussi son amour pour l'OM et le Vélodrome.

Les derniers échos indiquent qu'il va poursuivre l'aventure à Marseille, donc je suis plutôt confiant là-dessus, même si, dans le foot, tout peut aller très vite — on le sait trop bien.

Adrien Rabiot s'est lui exprimé dans un entretien pour La Gazzetta dello Sport.
Il a notamment été interrogé sur sa relation avec Allegri, qui vient de s'engager avec Milan, et sur la possibilité de le rejoindre là-bas.
Il a expliqué qu'il l'appréciait beaucoup, qu'il discuterait volontiers avec lui, tout en précisant qu'il était très content de s'être qualifié pour la Ligue des Champions avec l'OM.
Et que ce serait magnifique de la jouer au Vélodrome dans cette ferveur incroyable.

La Gazzetta nous apprend aussi que Rabiot aurait une clause dans son contrat lui permettant de partir contre 10 millions d'euros.
Un "gentleman agreement" qui ne permet pas un départ gratuit, mais pour un montant bien inférieur à sa valeur réelle.
Milan et Allegri pourraient donc se positionner et faire douter Rabiot, même si on espère qu'il s'inscrira dans la durée avec nous, pour faire de belles perfs en LDC.

Côté arrivées potentielles, l'OM apprécierait beaucoup le profil de Souffian El Karouani, d'après les infos d'Akim, relayées par La Tribune Olympienne.
Le latéral gauche de 24 ans évolue actuellement à Utrecht, aux Pays-Bas, avec qui il sort d'une saison très intéressante.
Titulaire indiscutable, avec deux buts et surtout 8 passes décisives, c'est un latéral très impactant dans ses montées, avec une bonne qualité de centre et de bons coups de pied arrêtés.
Il défend aussi correctement, avec des interceptions et des tacles bien sentis.

Son profil est vraiment intéressant : il peut jouer arrière gauche, piston gauche, et même milieu de terrain en cas de besoin.
Il est évalué autour de 9 millions d'euros par Transfermarkt, et il ne lui reste qu'un an de contrat avec son club, donc ça peut être une bonne opportunité.
Attention tout de même, d'autres clubs de Ligue 1 seraient aussi intéressés.

Je ne le vois pas forcément comme le futur titulaire à gauche, mais si Garcia s'en va — ou si le club décide de le laisser partir — El Karouani pourrait être une bonne doublure.

À noter aussi qu'il est international marocain, mais qu'il n'a plus été appelé chez les Lions de l'Atlas depuis le départ de Walid Regragui fin 2022.
Un transfert dans un club plus "up" pourrait l'aider à regagner sa place en sélection."""

    ,"test-reunion.mp3": """Comme un orchestre philharmonique, une réunion engage un ensemble de personnes qui doivent s'accorder pour produire quelque chose de cohérent. Le chef de projet donne le tempo et aide les participants à ne pas s'égarer, à rester dans le droit chemin de l'efficacité.

    — Bon, si vous voulez bien, commencez au quart de début.
    — Non mais, on n'a qu'à commencer, hein.
    — Mais qu'est-ce qui est fou ? C'est pas possible, ça…
    — Et c'était bien inscrit à l'accueil !
    — Moi je suis désolé. Si Catherine m'a... applaudie, je n'ai pas l'annonce.
    — Non, alors vous n'allez pas en rajouter non plus.
    — Mais vous êtes gonflée, vous ?
    — Mais quoi ?
    — Bon. Donc je vous ai dit que vous étiez gonflée.

    C'est réuni aujourd'hui parce qu'il est temps que nous fassions un point sur la chaîne de production des instruments pour l'orchestre de la cité.

    — Attends, j'ai jamais fait ma clé USB !
    — Tu m'as pas mené… c'est pas bête.
    — Bah quoi ? Non mais c'est mes photos de la Barbade ! Vous voulez les voir ?
    — Alors on va peut-être commencer, là.
    — Moi, ça m'était égal hein, j'ai déjà vu. Moi c'est pour vous.
    — Bon. Tout le monde est là ? Parfait. Il est l'heure. On commence.

    Bien. Vous avez tous reçu le dernier planning, le tableau de bord et le plan d'action. Parce que le concert a lieu en début d'année prochaine, et l'orchestre a besoin d'essences humaines — pardon, ressources humaines — au moins deux mois à l'avance pour pouvoir commencer les répétitions.

    — D'accord, question ?
    — Ah mais c'est nul, ça.
    — Vous me laissez tomber comme vous laissez la presse, hein !
    — Justement, c'est pas à ton tour d'amener cette semaine ?
    — Non mais moi je vois pas de quoi tu parles !
    — Ouais ouais c'est ça… radine, va.
    — Non mais je rêve !
    — C'est une personne dira ça, là !
    — Oui enfin…
    — Habib, dites-le là si vous avez monté une cabale au moins !
    — Tu files à temps morts en stop ? Tu vas ensuite ?

    Bien. Voici le plan d'action pour les semaines à venir. Apparemment, on a terminé la production des violons, mais il nous manque encore trois instruments pour Gauthier.

    — Quatre, neuf. Vous avez quelque chose à nous dire ?
    — Moi, en tant que responsable qualité, je vérifie juste si on fait les bonnes notes.
    — Mais moi je voulais toujours dire : les violons pour Gauthier, c'est pas des instruments comme les autres. Il nous faut un spécialiste. Et c'est pas avec les intérimaires qu'on m'a collés dans les pattes que je vais respecter les délais !

    — Non mais écoutez, c'est quand même pas compliqué.
    Ici, il y a la répartition des tâches par personne et par jour.
    Et dans cette colonne, il y a les solutions en cas de problème.
    Qu'est-ce que vous demandez de plus ?

    — Bah moi si je regarde le planning, je vois que j'ai pas besoin de revenir avant Pâques.
    Et puis, dans la troisième colonne, je vois pas bien si c'est des violons ou des guitares.
    — Vous allez pas me dire que c'est la faute de mon plan d'action ?
    Ça fait cinq semaines qu'on est sur la commande, vous allez pas me faire croire que vous découvrez ça aujourd'hui !

    Bon, reprenons. Il serait bon de faire un point sur l'avancée de la production des instruments.
    La question, c'est : est-ce qu'on peut tenir les délais négociés par l'orchestre ?

    Si on prend le planning…
    Toi, Francis, tu en es où avec la section cordes ?

    — C'est pas compliqué : selon la taille des employés, on leur donne les contrebasses ou les violons.
    — Oui alors là-dessus, j'ai eu des retours comme quoi il y avait des discriminations.
    — Bon, du scrimi d'assurance, on en a eu des mots, on en a eu des bas, mais c'est pas contre vous personnellement.
    — Faut pas le prendre pour l'habitude.
    — Oui, enfin, le résultat, c'est qu'on se retrouve avec des violons et des contrebasses qui ressemblent à des violas en scie.
    — Non mais c'est pas ça que je veux dire.
    Dans trois semaines, on doit pouvoir livrer douze violons.
    — Oui : douze violons, cinq altos, et trois contrebasses.

    Concrètement, on en est où ?
    Est-ce qu'on peut faire un premier contrôle qualité ?

    — Pour le contrôle, moi j'ai rien reçu.
    D'ailleurs, ça va être comme d'habitude : tout va arriver en même temps, au dernier moment.
    Et on va se retrouver à faire passer le contrôle qualité directement par le client.
    Et à se prendre tous les retours en service après-vente.

    — Non non non, on peut pas faire ça.
    À chaque fois, on doit rembourser les mécontents à cause des clauses de retard, on perd toute notre marge !
    — Il faut plus de luthiers !
    — Non ! On est stagiaires !
    — On va pas laisser faire ça.
    — Et à la fin, on se retrouve toujours avec des pipos à la place des flûtes.

    — Non mais alors là, je te l'envoie, je te la corrige, je te la remets en français.
    — Des luthiers.
    — Oui, en même temps, c'est pas la première fois que vous construisez un orchestre, vous auriez peut-être pu prévoir un petit peu.

    Parce que si je regarde le dernier tableau de bord, qu'est-ce que je vois ?
    Un très gros indicateur de risque sur la partie production. Comme par hasard, Francis !

    — Ben je voudrais bien vous y voir.
    Au départ, on commence par un casse-tête, on termine avec un philharmonique, et tout ça pour pas un sou de plus.
    Et à qui qu'on demande de tirer sur la corde ? À Bibi !
    — Oui, ben t'as aussi tiré sur la qualité.

    D'ailleurs, on se traîne une réputation… pas top top.
    — Ils vont pas nous faire un concert totale catastrophe, hein !

    Concrètement : est-ce qu'on peut avoir les instruments attendus ou non ?
    — Oui.
    — Oui.

    Bon. Et dans la gamme nouveau produit, vous avez réfléchi à ce que je vous avais demandé concernant les exigences de l'orchestre ?

    — On demande des cors.
    — La dernière réunion…
    — J'étais malade, moi, la dernière réunion ! J'ai pas bien compris, quand même.
    — Moi, j'étais à la Barbade d'ailleurs !
    — Non, non. C'est pas vrai.
    — S'il vous plaît. Moi je ne veux pas vous embêter, mais j'avoue que vous auriez pu nous envoyer un petit mail pour me le rappeler.
    — C'est là que moi j'avais pensé à ça.
    — Mais vous, votre truc, c'est pas la compta ? Plutôt les chiffres et tout ça ?
    — Qu'est-ce que vous y connaissez en instruments de musique ?

    — Non mais ce que je veux dire, c'est que ce n'est peut-être pas votre objectif premier de vous occuper de la partie recherche ?
    — Ce que je veux dire, c'est qu'on se fait une réunion d'avancement de projet… où personne n'a ramené de nouvelles idées !
    — Justement, j'avais pensé à…

    — Non, mais ça, c'est pas possible.
    — Il y a quelqu'un qui note ce qu'on fait, là ? Sinon, ça va être exactement la même chose la semaine prochaine.

    — J'ai envie de faire une voix de fin. Parce que…
    (jingle d'exaspération collectif)

    — Donc vous êtes en train de me dire que vous n'avez pas reçu les plans des flûtes ?
    — Ah mais si ! Je les ai, moi !
    Ça fait deux semaines que je les ai sur mon bureau, ces plans !

    — Si vous le dites pas, je peux pas deviner, moi !
    — Mais qu'est-ce que vous en avez à faire, vous, des plans de flûtes ?
    — Ce que Vincent veut dire, c'est que vous auriez pu le signaler plus tôt !
    — Oui enfin, c'est pas exactement ça, mais bon…

    Alors là, on vient perforer l'espace du lit, le passage de l'air se fait ici, dans ce sens, c'est particulier… voilà.

    Vous avez tout ce qu'il vous faut, en fait. Donc quoi ? Ce qui vous manque, c'est la motivation.

    Parce qu'en tant que chef de projet, je peux trouver des méthodes de travail plus expéditives.
    Ou alors c'est le patron qui va s'en charger. Et là, je vais pas vous dire que ce sera agréable.

    — Non non non…
    — Ce que Vincent veut dire, c'est que maintenant qu'on a déterminé les problèmes, on va pouvoir trouver les solutions.
    — Parce que ce planning-là, il n'est pas fait pour les chiens.
    Et quand on se réunit comme ça en fin de semaine, c'est pas pour jouer du pipeau sur tout votre truc.

    — Pardon, Francis.
    Mais s'il y a bien quelqu'un qui trinque avec du feu au-dessus…
    — Ah mais y a pas de mal à ce que tu l'utilises un peu !
    — Je sais pas… un bout de trique, c'est mon modèle technique.

    — Non, excusez-moi. Petit point de principe sur les échanges : je ne comprends pas bien ce que je fais dans une réunion technique.
    — On sait jamais. On aurait pu avoir des questions de budget.

    — En même temps, c'était pas à l'ordre du jour.
    Mais bon. Résultat : je suis venu pour rien.

    — Non pas que ce n'était pas passionnant, mais j'ai toutes les commandes à terminer.
    Alors, si vous permettez…

    — Non mais maintenant que vous êtes là, restez. Y a les tartelettes là, franchement !
    — Oui oui, c'est vrai. Dommage.

    Là où nous sommes les meilleurs, c'est pour la finition.
    Notre finition a la particularité de laisser un grand champ d'action au musicien.
    Elle permet à l'artiste de personnaliser son instrument, de trouver son son à lui."""
}

def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure le logging pour benchmark."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def get_ground_truth(audio_file: Path) -> Optional[str]:
    """Récupère le ground truth pour un fichier audio."""
    filename = audio_file.name
    if filename in GROUND_TRUTH_DATA:
        return GROUND_TRUTH_DATA[filename]

    # Essai sans extension
    stem = audio_file.stem
    for gt_name in GROUND_TRUTH_DATA.keys():
        if Path(gt_name).stem == stem:
            return GROUND_TRUTH_DATA[gt_name]

    return None

def format_benchmark_time(seconds: float) -> str:
    """Formate le temps de benchmark."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{seconds:.1f}s"

def transcribe_with_model(audio_path: Path, model: str, language: str = "fr") -> dict:
    """Transcrit avec un modèle Whisper spécifique."""
    logger = logging.getLogger(__name__)

    logger.info(f"🎤 Transcription {model}...")
    start_time = time.time()

    try:
        result = transcribe_meeting_audio(
            audio_path,
            model_size=model,
            language=language,
            temperature=0.0
        )

        if "error" in result:
            logger.error(f"❌ Erreur {model}: {result.get('message', 'Unknown')}")
            return {"error": result["error"], "text": "", "duration": 0}

        transcription_time = time.time() - start_time
        result["benchmark_time"] = transcription_time

        logger.info(f"✅ {model} terminé en {format_benchmark_time(transcription_time)}")
        return result

    except Exception as e:
        logger.error(f"❌ Exception {model}: {e}")
        return {"error": str(e), "text": "", "duration": 0}

def run_benchmark(audio_path: Path, models: List[str], language: str = "fr",
                 include_rouge: bool = True) -> Dict:
    """Lance le benchmark complet."""
    logger = logging.getLogger(__name__)

    # Validation audio
    if not validate_audio_path(audio_path):
        raise ValueError(f"Fichier audio invalide: {audio_path}")

    # Récupération ground truth
    ground_truth = get_ground_truth(audio_path)
    if not ground_truth:
        logger.warning(f"⚠️ Pas de ground truth pour {audio_path.name}")
        logger.info("💡 Benchmark transcription uniquement (pas de WER/ROUGE)")

    logger.info("🚀 BENCHMARK SUMMORA - MODÈLES WHISPER")
    logger.info(f"📁 Fichier: {audio_path.name}")
    logger.info(f"🤖 Modèles: {', '.join(models)}")
    logger.info(f"🎯 Ground truth: {'✅' if ground_truth else '❌'}")
    logger.info("-" * 60)

    # Transcriptions
    transcriptions = {}
    transcription_times = {}

    for model in models:
        result = transcribe_with_model(audio_path, model, language)
        transcriptions[model] = result.get("text", "")
        transcription_times[model] = result.get("benchmark_time", 0)

        # Log immédiat des résultats
        if result.get("error"):
            logger.error(f"💥 {model}: ÉCHEC ({result['error']})")
        else:
            word_count = result.get("word_count", 0)
            confidence = result.get("meeting_confidence", {}).get("meeting_confidence", 0)
            logger.info(f"📊 {model}: {word_count} mots, confiance {confidence:.3f}")

    # Métriques WER/ROUGE si ground truth disponible
    evaluation_reports = {}
    if ground_truth:
        logger.info("\n🔬 ÉVALUATION WER + ROUGE")
        logger.info("-" * 30)

        try:
            evaluator = create_business_evaluator()
            evaluation_reports = evaluator.compare_models(
                ground_truth, transcriptions, include_rouge
            )

            # Log des scores
            for model, report in evaluation_reports.items():
                wer = getattr(report.wer_result, 'business_wer', report.wer_result.score)
                rouge = report.rouge_result.score if include_rouge else 0
                logger.info(f"📈 {model}: WER {wer:.3f} | ROUGE {rouge:.3f} | Grade {report.business_quality_grade}")

        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
            evaluation_reports = {}

    return {
        "audio_file": str(audio_path),
        "ground_truth_available": bool(ground_truth),
        "models_tested": models,
        "transcriptions": transcriptions,
        "transcription_times": transcription_times,
        "evaluation_reports": evaluation_reports,
        "benchmark_timestamp": datetime.now().isoformat()
    }

def save_benchmark_results(results: Dict, output_file: Optional[str] = None) -> str:
    """Sauvegarde les résultats de benchmark."""
    if output_file:
        save_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(results["audio_file"]).stem
        save_path = Path('output/benchmarks') / f"benchmark_{audio_name}_{timestamp}.json"

    # Création du dossier si nécessaire
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Préparation des données pour JSON
    json_data = results.copy()

    # Conversion des objets MetricResult en dict
    if "evaluation_reports" in json_data:
        converted_reports = {}
        for model, report in json_data["evaluation_reports"].items():
            if hasattr(report, '__dict__'):
                converted_reports[model] = {
                    "composite_score": report.composite_score,
                    "business_quality_grade": report.business_quality_grade,
                    "wer_score": getattr(report.wer_result, 'business_wer', report.wer_result.score),
                    "rouge_score": report.rouge_result.score,
                    "recommendations": report.recommendations,
                    "processing_time": report.processing_time_total
                }
            else:
                converted_reports[model] = report
        json_data["evaluation_reports"] = converted_reports

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return str(save_path)

def print_benchmark_summary(results: Dict):
    """Affiche un résumé du benchmark."""
    print("\n" + "="*70)
    print("📊 RÉSUMÉ BENCHMARK SUMMORA")
    print("="*70)

    # Infos générales
    audio_file = Path(results["audio_file"])
    models = results["models_tested"]
    has_gt = results["ground_truth_available"]

    print(f"📁 Fichier testé    : {audio_file.name}")
    print(f"🤖 Modèles testés   : {', '.join(models)}")
    print(f"🎯 Ground truth     : {'✅ Disponible' if has_gt else '❌ Indisponible'}")
    print(f"⏱️  Benchmark le     : {results['benchmark_timestamp'][:19]}")

    # Résultats transcription
    print(f"\n📝 TEMPS DE TRANSCRIPTION")
    print("-" * 40)
    transcription_times = results["transcription_times"]
    for model in models:
        time_sec = transcription_times.get(model, 0)
        print(f"{model:8} : {format_benchmark_time(time_sec)}")

    # Résultats métriques si disponibles
    if has_gt and results["evaluation_reports"]:
        print(f"\n🔬 MÉTRIQUES WER + ROUGE")
        print("-" * 40)
        print(f"{'Modèle':<8} {'Score':<8} {'Grade':<6} {'WER':<8} {'ROUGE':<8}")
        print("-" * 40)

        eval_reports = results["evaluation_reports"]
        best_model = None
        best_score = -1

        for model in models:
            if model in eval_reports:
                report = eval_reports[model]
                score = report.get("composite_score", 0)
                grade = report.get("business_quality_grade", "N/A")
                wer = report.get("wer_score", 0)
                rouge = report.get("rouge_score", 0)

                print(f"{model:<8} {score:<8.3f} {grade:<6} {wer:<8.3f} {rouge:<8.3f}")

                if score > best_score:
                    best_score = score
                    best_model = model

        if best_model:
            print(f"\n🏆 MEILLEUR MODÈLE : {best_model} (score: {best_score:.3f})")

            # Recommandations du meilleur modèle
            best_report = eval_reports[best_model]
            recommendations = best_report.get("recommendations", [])
            if recommendations:
                print(f"\n💡 RECOMMANDATIONS:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec}")

    # Aperçu transcriptions
    print(f"\n📖 APERÇU TRANSCRIPTIONS")
    print("-" * 40)
    transcriptions = results["transcriptions"]
    for model in models:
        text = transcriptions.get(model, "")
        if text:
            preview = text[:100] + "..." if len(text) > 100 else text
            word_count = len(text.split())
            print(f"{model:8} ({word_count:4d} mots): {preview}")
        else:
            print(f"{model:8} (ÉCHEC)")

    print("="*70)

def main():
    """Point d'entrée principal du benchmark."""
    parser = argparse.ArgumentParser(
        description="Summora - Benchmark Modèles Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
python benchmark_models.py data/audio-om-mercato-test.mp3          # Tous modèles
python benchmark_models.py data/test-reunion.mp3 --models tiny base # Modèles spécifiques
python benchmark_models.py audio.mp3 --no-rouge --quiet           # WER uniquement
python benchmark_models.py audio.mp3 --aws-models                 # Medium + Large sur AWS
"""
    )

    # Arguments obligatoires
    parser.add_argument(
        "audio_file",
        type=str,
        help="Fichier audio à benchmarker"
    )

    # Configuration modèles
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["tiny", "base", "small"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="Modèles à tester (défaut: tiny base small)"
    )

    parser.add_argument(
        "--aws-models",
        action="store_true",
        help="Teste medium et large (nécessite GPU AWS)"
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Teste tous les modèles (tiny à large)"
    )

    # Configuration évaluation
    parser.add_argument(
        "--no-rouge",
        action="store_true",
        help="Désactive ROUGE (WER uniquement)"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default="fr",
        help="Langue de transcription (défaut: fr)"
    )

    # Options de sortie
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Fichier de sortie JSON"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne sauvegarde pas les résultats"
    )

    # Options système
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Mode silencieux"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    try:
        # Détermination des modèles à tester
        if args.all_models:
            models = ["tiny", "base", "small", "medium", "large"]
        elif args.aws_models:
            models = ["medium", "large"]
        else:
            models = args.models

        # Validation fichier audio
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            print(f"❌ Fichier audio introuvable: {audio_path}")
            return 1

        # Affichage info de démarrage (sauf mode quiet)
        if not args.quiet:
            print("🚀 SUMMORA - BENCHMARK MODÈLES WHISPER")
            print(f"📁 Fichier: {audio_path.name}")
            print(f"🤖 Modèles: {', '.join(models)}")
            print(f"🌍 Langue: {args.language}")
            if args.no_rouge:
                print("🔬 Métriques: WER uniquement")
            print("-" * 60)

        # Lancement du benchmark
        benchmark_start = time.time()

        results = run_benchmark(
            audio_path,
            models,
            args.language,
            include_rouge=not args.no_rouge
        )

        benchmark_total_time = time.time() - benchmark_start
        results["total_benchmark_time"] = benchmark_total_time

        # Sauvegarde des résultats
        if not args.no_save:
            saved_path = save_benchmark_results(results, args.output)
            if not args.quiet:
                logger.info(f"💾 Résultats sauvegardés: {saved_path}")

        # Affichage du résumé
        if not args.quiet:
            print_benchmark_summary(results)
            print(f"\n⏱️ Temps total benchmark: {format_benchmark_time(benchmark_total_time)}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
