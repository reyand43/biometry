from collections import Counter
from typing import List, Tuple

from domain.classifiers import BRISKClassifier, FASTClassifier, HarrisClassifier
from domain.utils.load_data import load

def recognition(image: List) -> Tuple[str, List]:
    data = load()

    classifiers = [
        HarrisClassifier(),
        BRISKClassifier(),
        FASTClassifier()
    ]

    X_train = [img for img, _ in data]
    y_train = [style for _, style in data]

    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    class_marks = Counter()
    features = []
    marks = []
    for classifier in classifiers:
        mark_with_feature = classifier.predict(image)
        class_marks[mark_with_feature[0][0]] += 1
        marks.append(mark_with_feature[0][0])
        features.append(mark_with_feature[0][1])

    mark = class_marks.most_common(1)[0][0]

    return mark, marks
