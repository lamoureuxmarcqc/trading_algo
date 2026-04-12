class BuffettInstitutional:

    def combine_scores(self, technical_score, quality_score):

        # 70% qualité / 30% marché
        return (quality_score * 0.7) + (technical_score * 0.3)
