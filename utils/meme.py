class Meme:
    def __init__(
        self, 
        meme_id: int,
        author_id: int, 
        path: str, 
        generated_round: int,
        description: str, 
        viewpoint: str,
        scores: dict = None,
        times_shared: int = 0,
        generation_json: str = None
    ):
        """
        Initialize a Meme object.

        Args:
            path (str): Path to the meme image.
            description (str): Description of the meme.
            viewpoint (str): Viewpoint of the meme creator.
        """
        self.meme_id = meme_id
        self.author_id = author_id
        self.path = path
        self.generated_round = generated_round

        self.description = description
        self.viewpoint = viewpoint
        self.generation_json = generation_json

        self.scores = scores if scores is not None else {}
        self.times_shared = times_shared

    def __repr__(self):
        return f"Meme(meme_id={self.meme_id}, author_id={self.author_id}, path='{self.path}', description='{self.description}', viewpoint='{self.viewpoint}', scores={self.scores}, times_shared={self.times_shared})"

    def _assign_score(
        self, 
        agent_id: str, 
        response: dict
    ):
        if agent_id in self.scores.keys():
            # raise ValueError(f"Agent {agent_id} has already scored this meme.")
            print(f"Agent {agent_id} has already scored this meme.")
        # Each meme will only be scored once by the agent
        else:
            self.scores[agent_id] = response

    def to_dict(self):
        return {
            "meme_id": self.meme_id,
            "author_id": self.author_id,
            "path": self.path,
            "generated_round": self.generated_round,
            "description": self.description,
            "viewpoint": self.viewpoint,
            "generation_json": self.generation_json,
            "scores": self.scores,
            "times_shared": self.times_shared
        }