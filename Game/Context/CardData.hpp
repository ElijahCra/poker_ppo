//
// Created by Elijah Crain on 12/31/25.
//

#ifndef POKER_CARDDATA_HPP
#define POKER_CARDDATA_HPP

namespace Game {
// Card data
struct CardData {
    inline static hand_indexer_t flopIndexer;
    inline static bool init = false;

    std::array<uint64_t, NUM_PLAYERS * NUM_ROUNDS> handHashes{};  // 2 players x 4 rounds
    // Raw card indices (0..51) for observation building.
    // Layout: [0..1] p0 hole, [2..3] p1 hole, [4..6] flop, [7] turn, [8] river
    std::array<uint8_t, 9> rawCards{};

    void initialize(std::mt19937& rng, const GameConfig& cfg) {
        if (!init) {
            // Static indexer is initialized once with the fixed {2,3,1,1}
            // round structure. GameConfig::validate() ensures cfg matches.
            constexpr uint8_t cardsperround[]{2, 3, 1, 1};
            hand_indexer_init(4, cardsperround, &flopIndexer);
            init = true;
        }
        // Shuffle only the cards still in play; dealt IDs stay in [0..51]
        // so the hand_indexer and TwoPlusTwo evaluator work unchanged.
        std::vector<uint8_t> deck = cfg.dealable_cards();
        std::ranges::shuffle(deck, rng);

        for (int i = 0; i < 9; ++i) rawCards[i] = deck[i];

        hand_indexer_state_t hand1indeces;
        hand_indexer_state_t hand2indeces;

        hand_indexer_state_init(&flopIndexer, &hand1indeces);
        hand_indexer_state_init(&flopIndexer, &hand2indeces);

        const uint8_t cardsp0[]{deck[0], deck[1]};
        const uint8_t cardsp1[]{deck[2], deck[3]};
        const uint8_t cardsflop[]{deck[4], deck[5], deck[6]};
        const uint8_t cardsturn[]{deck[7]};
        const uint8_t cardsriver[]{deck[8]};

        handHashes[0] = hand_index_next_round(&flopIndexer, cardsp0, &hand1indeces);
        handHashes[4] = hand_index_next_round(&flopIndexer, cardsp1, &hand2indeces);

        handHashes[1] = hand_index_next_round(&flopIndexer, cardsflop, &hand1indeces);
        handHashes[5] = hand_index_next_round(&flopIndexer, cardsflop, &hand2indeces);

        handHashes[2] = hand_index_next_round(&flopIndexer, cardsturn, &hand1indeces);
        handHashes[6] = hand_index_next_round(&flopIndexer, cardsturn, &hand2indeces);

        handHashes[3] = hand_index_next_round(&flopIndexer, cardsriver, &hand1indeces);
        handHashes[7] = hand_index_next_round(&flopIndexer, cardsriver, &hand2indeces);
    }
};

}

#endif //POKER_CARDDATA_HPP