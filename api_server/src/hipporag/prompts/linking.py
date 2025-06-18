def get_query_instruction(linking_method: str) -> str:
    """
    Maps a *linking method* to a natural-language retrieval instruction.

    :param linking_method: Strategy name used during retrieval.  
                           Supported values  
                           • ``"ner_to_node"``       – match a phrase against entity nodes.  
                           • ``"query_to_node"``     – match the full question against entity nodes.  
                           • ``"query_to_fact"``     – match the question against (subject, predicate, object) facts.  
                           • ``"query_to_sentence"`` – match the question against individual sentences.  
                           • ``"query_to_passage"``  – match the question against whole passages/documents.
    :return:              Human-readable instruction string.  
                          If *linking_method* is unrecognised, a generic
                          “retrieve relevant documents” instruction is returned.
    """
    instructions = {
        "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
        "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
        "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
        "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
        "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
    }
    default_instruction = (
        "Given a question, retrieve relevant documents that best answer the question."
    )
    return instructions.get(linking_method, default_instruction)