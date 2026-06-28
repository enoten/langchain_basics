text1 = '''Extremely disappointed with my recent iPhone purchase from Apple. The device constantly lags, and the battery life is abysmal,
barely lasting through the day. Despite the hefty price tag, the performance is far from satisfactory. Customer support has been unhelpful,
providing no viable solutions to address these persistant issues. This experience has left me regretting my decision to choose Apple,
and I expected much better from a company known for its premium products.'''

text2 = '''I recently purchased an iPhone from Apple, and it has been an absolute delight! The device runs smoothly, and the battery life is impressive, easily lasting throughout the day.
The price, though high, is justified by the excellent performance and top-notch customer support. I am thoroughly satisfied with my decision to choose Apple, and it reaffirms their reputation
for delivering premium products. Highly recommended for anyone seeking a reliable and high-performance smartphone'''

from transformers import pipeline
classifier = pipeline("text-classification")

import pandas as pd

# Analyze for text1
outputs1 = classifier(text1)
df1 = pd.DataFrame(outputs1)
print("Sentiment Analysis for text1:")
print(df1)

# Analyze for text1
outputs2 = classifier(text2)
df2 = pd.DataFrame(outputs2)
print("Sentiment Analysis for text2:")
print(df2)