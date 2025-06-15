import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple, Dict
from enum import Enum

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"  
    SPADES = "♠"

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        rank_str = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 
                   9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
        return f"{rank_str[self.rank.value]}{self.suit.value}"
    
    def __repr__(self):
        return str(self)

class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

class PokerHand:
    def __init__(self, cards: List[Card]):
        self.cards = sorted(cards, key=lambda x: x.rank.value, reverse=True)
        self.rank, self.value = self._evaluate_hand()
    
    def _evaluate_hand(self) -> Tuple[HandRank, List[int]]:
        ranks = [card.rank.value for card in self.cards]
        suits = [card.suit for card in self.cards]
        rank_counts = {}
        
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)
        
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True)
        
        if is_straight and is_flush:
            if min(ranks) == 10:
                return HandRank.ROYAL_FLUSH, [14]
            return HandRank.STRAIGHT_FLUSH, [max(ranks)]
        
        if counts == [4, 1]:
            return HandRank.FOUR_KIND, unique_ranks
        
        if counts == [3, 2]:
            return HandRank.FULL_HOUSE, unique_ranks
        
        if is_flush:
            return HandRank.FLUSH, sorted(ranks, reverse=True)
        
        if is_straight:
            return HandRank.STRAIGHT, [max(ranks)]
        
        if counts == [3, 1, 1]:
            return HandRank.THREE_KIND, unique_ranks
        
        if counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, unique_ranks
        
        if counts == [2, 1, 1, 1]:
            return HandRank.PAIR, unique_ranks
        
        return HandRank.HIGH_CARD, sorted(ranks, reverse=True)
    
    def _is_straight(self, ranks: List[int]) -> bool:
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) != 5:
            return False
        
        # 일반적인 스트레이트
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True
        
        # A-2-3-4-5 스트레이트 (로우 스트레이트)
        if unique_ranks == [2, 3, 4, 5, 14]:
            return True
        
        return False
    
    def __gt__(self, other):
        if self.rank.value != other.rank.value:
            return self.rank.value > other.rank.value
        return self.value > other.value

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        return self.cards.pop()

# DQN 네트워크 클래스 (trainAI.py와 동일)
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=4):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PokerGame:
    def __init__(self, ai_model_path: str = None):
        self.deck = Deck()
        self.player_chips = 1000
        self.ai_chips = 1000
        self.pot = 0
        self.community_cards = []
        self.player_hand = []
        self.ai_hand = []
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        # AI 모델 로드
        self.ai_model = None
        if ai_model_path:
            try:
                # DQN 체크포인트 로드
                checkpoint = torch.load(ai_model_path, map_location='cpu')
                
                # 체크포인트에서 모델 정보 추출
                input_dim = checkpoint.get('input_dim', 60)
                action_num = checkpoint.get('action_num', 4)
                
                # DQN 모델 생성
                self.ai_model = SimpleDQN(input_dim, output_dim=action_num)
                
                # 가중치 로드
                self.ai_model.load_state_dict(checkpoint['q_net_state_dict'])
                self.ai_model.eval()
                
                print(f"✅ DQN 모델을 성공적으로 로드했습니다: {ai_model_path}")
                print(f"입력 차원: {input_dim}, 액션 수: {action_num}")
                
            except Exception as e:
                print(f"❌ AI 모델 로드 실패: {e}")
                print("기본 랜덤 AI를 사용합니다.")
                self.ai_model = None
        else:
            print("AI 모델 경로가 제공되지 않았습니다. 기본 랜덤 AI를 사용합니다.")
    
    def start_new_hand(self):
        self.deck.reset()
        self.community_cards = []
        self.player_hand = []
        self.ai_hand = []
        self.pot = 0
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        # 블라인드 베팅
        small_blind = 10
        big_blind = 20
        
        self.player_chips -= small_blind
        self.ai_chips -= big_blind
        self.pot = small_blind + big_blind
        self.player_bet = small_blind
        self.ai_bet = big_blind
        self.current_bet = big_blind
        
        # 카드 배분
        for _ in range(2):
            self.player_hand.append(self.deck.deal_card())
            self.ai_hand.append(self.deck.deal_card())
    
    def get_game_state_vector(self) -> np.ndarray:
        """게임 상태를 벡터로 변환 (DQN 모델 입력용)"""
        # DQN 모델의 입력 차원에 맞춰 조정
        state = np.zeros(60)  # 기본 크기, 실제 모델에 따라 조정 필요
        
        try:
            # AI 핸드 인코딩
            for i, card in enumerate(self.ai_hand[:2]):
                # 슈트 원핫 인코딩
                suit_idx = ["♥", "♦", "♣", "♠"].index(card.suit.value)
                state[i*4 + suit_idx] = 1.0
                # 랭크 정규화
                state[8 + i] = card.rank.value / 14.0
            
            # 커뮤니티 카드 인코딩
            for i, card in enumerate(self.community_cards[:5]):
                suit_idx = ["♥", "♦", "♣", "♠"].index(card.suit.value)
                state[10 + i*4 + suit_idx] = 1.0
                state[30 + i] = card.rank.value / 14.0
            
            # 게임 정보
            state[35] = min(self.pot / 2000.0, 1.0)
            state[36] = min(self.ai_chips / 2000.0, 1.0)
            state[37] = min(self.player_chips / 2000.0, 1.0)
            state[38] = min(self.current_bet / 200.0, 1.0)
            state[39] = min(self.ai_bet / 200.0, 1.0)
            state[40] = min(self.player_bet / 200.0, 1.0)
            state[41] = len(self.community_cards) / 5.0
            
        except Exception as e:
            print(f"상태 벡터 생성 중 오류: {e}")
        
        return state
    
    def ai_action(self) -> Tuple[str, int]:
        """AI의 행동 결정"""
        if self.ai_model is None:
            # 랜덤 AI
            actions = ['fold', 'call', 'raise']
            action = random.choice(actions)
            
            if action == 'fold':
                return 'fold', 0
            elif action == 'call':
                call_amount = max(0, self.current_bet - self.ai_bet)
                return 'call', min(call_amount, self.ai_chips)
            else:  # raise
                min_raise = max(20, self.current_bet - self.ai_bet + 20)
                max_raise = self.ai_chips
                if min_raise <= max_raise:
                    raise_amount = random.randint(min_raise, min(max_raise, min_raise + 100))
                    return 'raise', raise_amount
                else:
                    return 'call', min(self.current_bet - self.ai_bet, self.ai_chips)
        
        # 학습된 DQN 모델 사용
        try:
            state = self.get_game_state_vector()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.ai_model(state_tensor)
                action_idx = torch.argmax(q_values, dim=1).item()
            
            # DQN 액션을 포커 액션으로 매핑
            # 0: fold, 1: call/check, 2: small raise, 3: big raise
            if action_idx == 0:
                return 'fold', 0
            elif action_idx == 1:
                call_amount = max(0, self.current_bet - self.ai_bet)
                return 'call', min(call_amount, self.ai_chips)
            else:  # raise (action_idx 2 or 3)
                min_raise = max(20, self.current_bet - self.ai_bet + 20)
                max_raise = self.ai_chips
                
                if min_raise > max_raise:
                    # 레이즈할 수 없으면 콜
                    call_amount = max(0, self.current_bet - self.ai_bet)
                    return 'call', min(call_amount, self.ai_chips)
                
                # 액션에 따른 레이즈 크기 결정
                if action_idx == 2:  # small raise
                    raise_amount = min_raise + (max_raise - min_raise) // 4
                else:  # big raise
                    raise_amount = min_raise + (max_raise - min_raise) // 2
                
                raise_amount = min(raise_amount, max_raise)
                return 'raise', max(min_raise, raise_amount)
                
        except Exception as e:
            print(f"AI 행동 결정 중 오류: {e}")
            # 오류 시 랜덤 행동
            actions = ['fold', 'call']
            action = random.choice(actions)
            if action == 'fold':
                return 'fold', 0
            else:
                call_amount = max(0, self.current_bet - self.ai_bet)
                return 'call', min(call_amount, self.ai_chips)
    
    def deal_flop(self):
        """플롭 (처음 3장의 커뮤니티 카드) 딜"""
        for _ in range(3):
            self.community_cards.append(self.deck.deal_card())
    
    def deal_turn(self):
        """턴 (4번째 커뮤니티 카드) 딜"""
        self.community_cards.append(self.deck.deal_card())
    
    def deal_river(self):
        """리버 (5번째 커뮤니티 카드) 딜"""
        self.community_cards.append(self.deck.deal_card())
    
    def get_best_hand(self, hole_cards: List[Card]) -> PokerHand:
        """7장 카드에서 최고의 5장 조합 찾기"""
        all_cards = hole_cards + self.community_cards
        best_hand = None
        
        from itertools import combinations
        for combo in combinations(all_cards, 5):
            hand = PokerHand(list(combo))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        
        return best_hand
    
    def determine_winner(self) -> str:
        """승자 결정"""
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        
        if player_best > ai_best:
            return "player"
        elif ai_best > player_best:
            return "ai"
        else:
            return "tie"
    
    def display_game_state(self, hide_ai_cards=True):
        """게임 상태 출력"""
        print("\n" + "="*50)
        print(f"팟: {self.pot}칩")
        print(f"현재 베팅: {self.current_bet}칩")
        print(f"플레이어 칩: {self.player_chips}칩 (베팅: {self.player_bet}칩)")
        print(f"AI 칩: {self.ai_chips}칩 (베팅: {self.ai_bet}칩)")
        print("-"*50)
        
        if hide_ai_cards:
            print(f"AI 핸드: [?, ?]")
        else:
            print(f"AI 핸드: {self.ai_hand}")
        
        print(f"플레이어 핸드: {self.player_hand}")
        
        if self.community_cards:
            print(f"커뮤니티 카드: {self.community_cards}")
        
        print("="*50)
    
    def betting_round(self) -> bool:
        """베팅 라운드 진행. True: 계속, False: 핸드 종료"""
        while True:
            # 플레이어 턴
            self.display_game_state()
            
            need_to_call = self.current_bet - self.player_bet
            if need_to_call > 0:
                print(f"\n콜하려면 {need_to_call}칩이 필요합니다.")
            
            print("\n행동을 선택하세요:")
            print("1. 폴드")
            print("2. 콜" if need_to_call > 0 else "2. 체크")
            print("3. 레이즈")
            
            try:
                choice = int(input("선택 (1-3): "))
                
                if choice == 1:  # 폴드
                    print("플레이어가 폴드했습니다.")
                    self.ai_chips += self.pot
                    return False
                
                elif choice == 2:  # 콜/체크
                    if need_to_call > 0:
                        call_amount = min(need_to_call, self.player_chips)
                        self.player_chips -= call_amount
                        self.player_bet += call_amount
                        self.pot += call_amount
                        print(f"플레이어가 {call_amount}칩을 콜했습니다.")
                    else:
                        print("플레이어가 체크했습니다.")
                    break
                
                elif choice == 3:  # 레이즈
                    min_raise = max(20, need_to_call + 20)
                    max_raise = self.player_chips
                    
                    if min_raise > max_raise:
                        print("레이즈할 수 없습니다. 올인하거나 콜하세요.")
                        continue
                    
                    raise_amount = int(input(f"레이즈 금액 ({min_raise}-{max_raise}): "))
                    
                    if raise_amount < min_raise or raise_amount > max_raise:
                        print("잘못된 금액입니다.")
                        continue
                    
                    self.player_chips -= raise_amount
                    self.player_bet += raise_amount
                    self.pot += raise_amount
                    self.current_bet = self.player_bet
                    print(f"플레이어가 {raise_amount}칩을 레이즈했습니다.")
                    break
                
                else:
                    print("잘못된 선택입니다.")
                    continue
                    
            except ValueError:
                print("숫자를 입력하세요.")
                continue
        
        # AI 턴
        need_to_call = self.current_bet - self.ai_bet
        
        if need_to_call == 0:
            print("AI가 체크했습니다.")
            return True
        
        ai_action, ai_amount = self.ai_action()
        
        if ai_action == 'fold':
            print("AI가 폴드했습니다.")
            self.player_chips += self.pot
            return False
        
        elif ai_action == 'call':
            actual_amount = min(ai_amount, self.ai_chips, need_to_call)
            self.ai_chips -= actual_amount
            self.ai_bet += actual_amount
            self.pot += actual_amount
            print(f"AI가 {actual_amount}칩을 콜했습니다.")
            return True
        
        else:  # raise
            self.ai_chips -= ai_amount
            self.ai_bet += ai_amount
            self.pot += ai_amount
            self.current_bet = self.ai_bet
            print(f"AI가 {ai_amount}칩을 레이즈했습니다.")
            
            # 플레이어가 다시 응답해야 함
            return self.betting_round()
    
    def play_hand(self):
        """한 핸드 플레이"""
        self.start_new_hand()
        
        # 프리플롭 베팅
        if not self.betting_round():
            return
        
        # 플롭
        self.deal_flop()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 턴
        self.deal_turn()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 리버
        self.deal_river()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 쇼다운
        print("\n" + "="*20 + " 쇼다운 " + "="*20)
        self.display_game_state(hide_ai_cards=False)
        
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        
        print(f"\n플레이어 베스트 핸드: {player_best.cards} ({player_best.rank.name})")
        print(f"AI 베스트 핸드: {ai_best.cards} ({ai_best.rank.name})")
        
        winner = self.determine_winner()
        
        if winner == "player":
            print("\n🎉 플레이어 승리!")
            self.player_chips += self.pot
        elif winner == "ai":
            print("\n🤖 AI 승리!")
            self.ai_chips += self.pot
        else:
            print("\n🤝 무승부!")
            self.player_chips += self.pot // 2
            self.ai_chips += self.pot // 2
    
    def play_game(self):
        """메인 게임 루프"""
        print("🃏 포커 AI 대전에 오신 것을 환영합니다! 🃏")
        print("텍사스 홀덤 포커로 AI와 1대1 대결을 펼치세요!")
        
        while self.player_chips > 0 and self.ai_chips > 0:
            print(f"\n현재 칩 상황 - 플레이어: {self.player_chips}, AI: {self.ai_chips}")
            
            play_again = input("\n새 핸드를 시작하시겠습니까? (y/n): ").lower()
            if play_again != 'y':
                break
            
            self.play_hand()
        
        # 게임 종료
        print("\n" + "="*50)
        print("게임 종료!")
        if self.player_chips > self.ai_chips:
            print("🎉 축하합니다! 플레이어가 최종 승리했습니다!")
        elif self.ai_chips > self.player_chips:
            print("🤖 AI가 최종 승리했습니다!")
        else:
            print("🤝 무승부입니다!")
        
        print(f"최종 칩 - 플레이어: {self.player_chips}, AI: {self.ai_chips}")

# 사용 예시
if __name__ == "__main__":
    # AI 모델 파일 경로 지정
    model_path = "final_poker_model.pth"  # 또는 "poker_model_500.pth" 등
    
    # 게임 시작
    game = PokerGame(model_path)
    game.play_game()