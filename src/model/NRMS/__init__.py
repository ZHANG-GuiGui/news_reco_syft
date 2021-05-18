import torch
from model.NRMS.news_encoder import NewsEncoder
from model.NRMS.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class NRMS(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, pretrained_word_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self,
                x,
                #candidate_news=None,
                #clicked_news=None
                ):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """
        # batch_size, 1 + K, word_embedding_dim

        #candidate_news_vector = torch.stack(
        #   [self.news_encoder(x) for x in candidate_news], dim=1)
        # candidate_news [batch_size, 1+K, length_title]
        #print(candidate_news.size())
        candidate_news = x[:,:3,:]
        clicked_news = x[:,3:,:]
        #candidate_news = candidate_news.transpose(0,1)
        #print(candidate_news.size())
        #print("*********************")
        #print(candidate_news.size())
        #print("*********************")
        candidate_news_vector = torch.stack(
            [self.news_encoder(candidate_news[:,i,:]) for i in range(3)],
            dim=1
        )
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        #clicked_news_vector = torch.stack(
        #    [self.news_encoder(x) for x in clicked_news], dim=1)
        #clicked_news = clicked_news.transpose(0,1)
        #size = clicked_news.size(0)
        clicked_news_vector = torch.stack(
            [self.news_encoder(clicked_news[:, i, :]) for i in range(50)],
            dim=1
        )
        #candidate_news_vector = self.news_encoder(candidate_news)
        #clicked_news_vector = self.news_encoder(clicked_news)
        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
