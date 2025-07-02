MEME_TEMPLATE_CHECK_SYSTEM_PROMPT = """You are an expert in meme template verification.
Your responses should reflect this expertise, helping to identify whether the example meme correctly follows the template meme."""

MEME_TEMPLATE_CHECK_USER_PROMPT = """You are provided with two memes. The first is the template meme; the second is an example meme.

## Task
- Template Text: Identify all texts in the template meme.
- Example Text: Identify all texts in the example meme.
- Match:
    - If the background of the example meme is different from the template meme, respond "False".
    - Determine whether the text placement in the example meme matches the placeholder positions in the template.
        - If the text placement in the example meme matches the placeholder positions in the template, respond "True".
        - If the text placement deviates (e.g., misplaced, moved or extra text at other locations), respond "False".

Response with the following JSON format:
{{
    template_text: [str]
    example_text: [str]
    match: bool
}}"""

EXPLAIN_MEME_SYSTEM_PROMPT = """You are an expert in internet memes and digital culture. 
Your responses should reflect this expertise, providing insightful analysis of meme structures, humor mechanics, and online communication trends.
"""

EXPLAIN_MEME_USER_PROMPT_WITH_ABOUT = """Meme background knowledge: ```{about}```. 
You are provided with background information about a meme template, and several example memes made using this template.

## Task
- Template Functionality: Clearly explain how the meme template functions in general. 
- Text Functionality: Explain how each numbered text overlay: {numbered_texts}; should be used in the meme. Focus on the general meaning or communicative function of the text.

## Requirements
- Focus strictly on the broad functions of the meme elements and the general interplay between the image and the text. 
- Avoid discussing any details specific to individual example memes.

Response in the following JSON format: 
{{
    template_functionality: str, 
    text_1: str, 
    text_2: str,
    ...
}}"""

EXPLAIN_MEME_USER_PROMPT_WITHOUT_ABOUT = """You are provided a meme template, and several example memes made using this template.

## Task
- Template Functionality: Clearly explain how the meme template functions in general. 
- Text Functionality: Explain how each numbered text overlay: {numbered_texts}; should be used in the meme. Focus on the general meaning or communicative function of the text.

## Requirements
- Focus strictly on the broad functions of the meme elements and the general interplay between the image and the text. 
- Avoid discussing any details specific to individual example memes.

Response in the following JSON format: 
{{
    template_functionality: str, 
    text_1: str, 
    text_2: str,
    ...
}}."""

PERSONA_PROMPT = {
    "fun": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You like to be clownish. You like to tease your friends in a funny way. You like to make jests and to be silly. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "irony": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You can converse with close friends in a way that only you all know what is meant, but outsiders don't sense that it is merely irony. If you say something that is ironic, there is always someone in your group who understands it, and others who don't. Your irony confuses those who don't understand it, as you and your close friends uphold what you all really mean. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "wit": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You have a sharp wit and intellect and can tell stories with many punch lines. You surprise others with funny remarks and accurate judgments of current issues, which occur to you spontaneously. Your wit and astute mind help you to be quick witted. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "sarcasm": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. Biting mockery suits you. You are a sharp-tongued detractor. You have a bitter, biting kind of mockery at your disposal, which you express both directly and indirectly (e.g., by means of irony). Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "humor": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. On a large and small scale, the world is not perfect, but with a humorous outlook on the world you can amuse yourself at the adversities of life. You accept the imperfection of human beings and your everyday life often gives you the opportunity to smile benevolently about it. Humor is suitable for arousing understanding and sympathy for imperfections and the human condition. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "satire": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You parody people's bad habits to fight the bad and foolish behavior. You caricature your fellow humans' wrongdoings in a funny way to gently urge them to change. You like to ridicule moral badness to induce or increase a critical attitude in other people. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "nonsense": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You like nonsensical humor. Absurdities amuse you. You like humor that flies in the face of logic. Your responses should faithfully reflect the attributes and characteristics of this persona.",
    "cynicism": "Adopt the persona of a Singaporean social media user who is knowledgeable in Singaporean meme culture and adept in making hilarious memes with Singaporean references. You disdain some moral norms and view them cynically, although you don't lack a sense of moral values in general. You have a cynical attitude towards some common norms and moral concepts; you don't believe in them and mostly find them ridiculous. You tend to show no reverence for certain moral concepts and ideals, but only scorn and derision. Your responses should faithfully reflect the attributes and characteristics of this persona."
}

STRATEGY_PROMPT = {
    "caption": "Article: {summarized_article}\n{viewpoint}Meme template information: {information}\nInstructions: {inspiration}Creatively suggest short overlay text(s) ({numbered_texts}) for the meme template such that the resulting meme hilariously but clearly expresses the stated viewpoint.\nConstraint: Answer as a valid JSON strictly with the following keys: {numbered_texts}.",
    "objlab": "Article: {summarized_article}\n{viewpoint}Instructions: {inspiration}Creatively suggest and describe a highly original object-labeling meme that hilariously but clearly expresses the stated viewpoint. Be creative and do not rely on existing meme templates. Object labeling memes comprise images where the text does not have a default position. Instead, it is placed upon objects or individuals and is used as a means of labeling them. Each act of labeling creates a metaphorical relationship between the object or person in the image and the object or person described in the label.\nConstraint: Answer without any explanation or preamble, in the strict format of \'Generate an object-labeling meme showing {{image_description}}. {{object1}} in the image is labeled \"{{label1}},\" {{object2}} in the image is labeled \"{{label2}}\" … \' Here, {{image_description}}, {{object1}}, {{label1}} and so on are all placeholders that must be replaced with appropriate words. There can be two or three object-label pairs. Keep the labels concise.",
    "multipanel": "Article: {summarized_article}\n{viewpoint}Instructions: {inspiration}Creatively suggest and describe a highly original four panels meme, and any accompanying texts within each of the four panels, that hilariously but clearly expresses the stated viewpoint. Be creative and do not rely on existing meme templates. A multipanel meme features consecutive images, combined to create a narrative that is dependent on movement through time or cause and effect. The images cannot be reordered.\nConstraint: Answer without any explanation or preamble, in the strict format of \'Generate a four-panel meme. The top-left panel shows {{top_left_description}}. The top-right panel shows {{top_right_description}}. The bottom-left panel shows {{bottom_left_description}}. The bottom-right panel shows {{bottom_right_description}}.\' Here, {{top_left_description}}, {{top_right_description}}, {{bottom_left_description}} and {{bottom_right_description}} are all placeholders that must be replaced with appropriate words. For each panel, where text should be added, either as accompanying text or to express dialogue, clearly state so. Keep your answer concise.",
    "imgmacro": "Article: {summarized_article}\n{viewpoint}Instructions: {inspiration}Creatively suggest and describe a highly original image-macro meme that hilariously but clearly expresses the stated viewpoint. Be creative and do not rely on existing meme templates. The primary features of image macros are the text font and the placement. The text predominantly occupies the upper and the lower part of the image, and the font used is the Impact Font with black border and white capital letters.\nConstraint: Answer without any explanation or preamble, in the strict format of \'Generate an image-macro meme showing {{image_description}}. The top text, in Impact font, on the meme says \"{{top_text}}.\" The bottom text, in Impact font, on the meme says \"{{bottom_text}}.\"\' Here, {{image_description}}, {{top_text}} and {{bottom_text}} are all placeholders that must be replaced with appropriate words. Keep the text overlays concise.",
    "edit": "Article: {summarized_article}\n{viewpoint}Instructions: Creatively suggest and describe edit(s) that will make the meme even more hilarious and better express the stated viewpoint. The edit(s) can involve, but is not limited to, replacing or adding objects to include Singaporean logo(s) and/or other Singaporean social or cultural symbol(s), and/or other object(s) unique to Singapore. You may also suggest replacing the text to better express the stated viewpoint.\nConstraint: Strictly describe the edit(s) without any explanation or any preamble.",
}

STRATEGY_DESCRIPTION_PROMPT = {
    "caption": "meme template captioner",
    "objlab": "object-labeling meme creator",
    "multipanel": "multipanel meme creator",
    "imgmacro": "image-macro meme creator",
    "edit": "meme editor"
}

ARTICLE_SELECTION_SYSTEM_PROMPT = """You are a creative social media user tasked to analyze several articles, select 1 article that are must be in Singapore context while you think best fits to generate memes and provide a summary for that article.

The user will provide the articles in the following JSON format:
{
    "Article 1's title": "content of article",
    "Article 2's title": "content of article",
    ...
}

Task:
1. Analyze the articles and select 1 article that is relevant to Singapore context and best fits to generate memes.
2. Provide a summary of the selected article.

Output your choice in the following JSON format:
{
    "title": "Selected article's title",
    "summary": "Summary of the selected article"
}

No additional output required."""

ANALYZE_ARTICLE_VIEWPOINT = """Definitions:
\"\"\"
Target: A target may refer to one concrete, tangible entity or one abstract subject that was commented on in the meme. A target can be an individual, an organization, a community, a society, a government policy, a movement, a product, etc., and can be expressed as a named entity, a common noun, or a multi-word term.
Aspect: An aspect is one characteristic, attribute, or feature of the target of the meme.
Opinion: An opinion is an evaluation or attitude toward the aspect of the target of the meme.
Sentiment: The sentiment is the polarity of the opinion, either \"positive,\" or \"neutral,\" or \"negative.\"
Viewpoint: A viewpoint comprises a target of the meme, an aspect of the target referenced, an opinion the meme expresses toward the aspect, and the sentiment polarity of the opinion.
\"\"\"

Instructions: Suggest a set of three one-sentence viewpoints relevant to the article that meme creators could espouse in their memes after reading the article. The targets and the aspects must be exactly identical, while the opinions and sentiments must differ. The viewpoints must contain exactly one viewpoint of positive sentiment, one of negative sentiment, and one of neutral sentiment.
Constraint: Strictly answer as a valid JSON, with the keys, \"viewpoint1a\" and \"viewpoint1b\" and \"viewpoint1c\". Each viewpoint must strictly follow the format of "The meme views {target noun or noun phrase} with a {sentiment} sentiment because its/his/her/their {aspect noun or noun phrase} is/are seen as {opinion adjective or adjective phrase}."""

MEME_EVALUATION_PROMPT = """As a {strategy_desc}, you are tasked to evaluate a meme that had been shared for {number_of_shares} times. 

{viewpoint}

Task:
1. On a scale of 1 to 7, where \n1 = \"strongly disagree\", \n2 = \"disagree\", \n3 = \"slightly disagree\", \n4 = \"neither disagree nor agree\", \n5 = \"slightly agree\", \n6 = \"agree\", \n7 = \"strongly agree\". \nRate this statement: I will \"Share\" this meme on social media.\nConstraint: Strictly return the numerical rating only.
2. Provide a brief explanation of your rating in no more than 50 words.

Response in the following JSON format:
{{
    "score": int,
    "explanation": str
}}"""