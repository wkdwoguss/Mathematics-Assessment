import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# RLCard 설치: pip install rlcard
try:
    import rlcard
    from rlcard.agents import RandomAgent
    from rlcard.envs.registration import make
    print("✅ RLCard 라이브러리 로드 성공")
except ImportError as e:
    print("❌ RLCard 설치 필요: pip install rlcard")
    raise e

class SimpleDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=4):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CustomDQNAgent:
    def __init__(self, input_dim, action_num, lr=0.001, device=None):
        # input_dim을 직접 받아서 정수로 보장
        self.input_dim = int(input_dim)
        self.action_num = action_num
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"에이전트 초기화 - 입력 차원: {self.input_dim}, 액션 수: {action_num}")
        
        # 네트워크 초기화
        self.q_net = SimpleDQN(self.input_dim, output_dim=action_num).to(self.device)
        self.target_net = SimpleDQN(self.input_dim, output_dim=action_num).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 하이퍼파라미터
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.memory_size = 10000
        
        # 경험 리플레이 버퍼
        self.memory = deque(maxlen=self.memory_size)
        
        # 타겟 네트워크 초기화
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.train_step = 0
        self.update_target_every = 100
    
    def feed(self, ts):
        """RLCard 인터페이스에 맞춘 경험 저장"""
        try:
            if len(ts) >= 4:
                if len(ts) == 4:
                    state, action, reward, next_state = ts
                    done = False
                else:
                    state, action, reward, next_state, done = ts[:5]
                
                self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"feed 오류: {e}")
    
    def step(self, state):
        """액션 선택 - RLCard 호환 (훈련용)"""
        return self._select_action(state, use_epsilon=True)
    
    def eval_step(self, state):
        """평가용 액션 선택 - RLCard 호환 (평가용, 탐험 없음)"""
        return self._select_action(state, use_epsilon=False)
    
    def _select_action(self, state, use_epsilon=True):
        """내부 액션 선택 로직"""
        try:
            # RLCard 환경에서 state는 딕셔너리 형태
            if isinstance(state, dict):
                obs = state['obs']
                legal_actions = state['legal_actions']
            else:
                obs = state
                legal_actions = list(range(self.action_num))
            
            # 상태 검증
            if obs is None:
                return int(np.random.choice(legal_actions))
            
            # 탐험 vs 활용 (평가 시에는 탐험 안함)
            if use_epsilon and np.random.random() <= self.epsilon:
                return int(np.random.choice(legal_actions))
            
            # 상태를 텐서로 변환 - 고정 차원으로
            try:
                if isinstance(obs, (list, np.ndarray)):
                    obs_array = np.array(obs, dtype=np.float32).flatten()
                    # 고정 차원으로 맞춤
                    if len(obs_array) >= self.input_dim:
                        obs_array = obs_array[:self.input_dim]
                    else:
                        padded_obs = np.zeros(self.input_dim, dtype=np.float32)
                        padded_obs[:len(obs_array)] = obs_array
                        obs_array = padded_obs
                    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
                else:
                    obs_array = np.zeros(self.input_dim, dtype=np.float32)
                    obs_array[0] = float(obs)
                    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
            except Exception as e:
                print(f"상태 변환 오류: {e}, obs: {obs}")
                return int(np.random.choice(legal_actions))
            
            # Q값 계산
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)
            
            # 유효한 액션만 고려
            if len(legal_actions) < self.action_num:
                masked_q_values = q_values.clone()
                for i in range(self.action_num):
                    if i not in legal_actions:
                        masked_q_values[0][i] = -float('inf')
                q_values = masked_q_values
            
            action = q_values.cpu().data.numpy().argmax()
            
            # numpy.int64를 Python int로 변환
            action = int(action)
            
            # 유효한 액션인지 확인
            if action not in legal_actions:
                action = int(np.random.choice(legal_actions))
            
            return action
            
        except Exception as e:
            print(f"액션 선택 오류: {e}")
            return int(np.random.choice(legal_actions)) if 'legal_actions' in locals() else 0
    
    def train(self):
        """네트워크 훈련"""
        if len(self.memory) < self.batch_size:
            return 0
            
        try:
            # 배치 샘플링
            batch = random.sample(self.memory, self.batch_size)
            
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for transition in batch:
                try:
                    state, action, reward, next_state, done = transition
                    
                    # 상태 처리
                    if isinstance(state, dict):
                        state_obs = state['obs']
                    else:
                        state_obs = state
                        
                    if isinstance(next_state, dict):
                        next_state_obs = next_state['obs']
                    else:
                        next_state_obs = next_state
                    
                    # None 체크
                    if state_obs is None or next_state_obs is None:
                        continue
                    
                    # 배열 변환 - 고정 차원으로
                    if isinstance(state_obs, (list, np.ndarray)):
                        state_obs = np.array(state_obs, dtype=np.float32).flatten()
                        if len(state_obs) >= self.input_dim:
                            state_obs = state_obs[:self.input_dim]
                        else:
                            padded_state = np.zeros(self.input_dim, dtype=np.float32)
                            padded_state[:len(state_obs)] = state_obs
                            state_obs = padded_state
                    else:
                        temp_obs = np.zeros(self.input_dim, dtype=np.float32)
                        temp_obs[0] = float(state_obs)
                        state_obs = temp_obs
                        
                    if isinstance(next_state_obs, (list, np.ndarray)):
                        next_state_obs = np.array(next_state_obs, dtype=np.float32).flatten()
                        if len(next_state_obs) >= self.input_dim:
                            next_state_obs = next_state_obs[:self.input_dim]
                        else:
                            padded_next_state = np.zeros(self.input_dim, dtype=np.float32)
                            padded_next_state[:len(next_state_obs)] = next_state_obs
                            next_state_obs = padded_next_state
                    else:
                        temp_next_obs = np.zeros(self.input_dim, dtype=np.float32)
                        temp_next_obs[0] = float(next_state_obs)
                        next_state_obs = temp_next_obs
                    
                    states.append(state_obs)
                    actions.append(int(action))
                    rewards.append(float(reward))
                    next_states.append(next_state_obs)
                    dones.append(bool(done))
                    
                except Exception as e:
                    print(f"배치 처리 오류: {e}")
                    continue
            
            if len(states) == 0:
                return 0
            
            # 텐서 변환
            try:
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.BoolTensor(dones).to(self.device)
            except Exception as e:
                print(f"텐서 변환 오류: {e}")
                print(f"states 형태 확인: {[s.shape if hasattr(s, 'shape') else type(s) for s in states[:3]]}")
                return 0
            
            # Q 값 계산
            current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # 손실 계산 및 역전파
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()
            
            # 타겟 네트워크 업데이트
            self.train_step += 1
            if self.train_step % self.update_target_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            
            # 입실론 감소
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss.item()
            
        except Exception as e:
            print(f"훈련 중 오류: {e}")
            return 0
    
    def save_model(self, path):
        """모델 저장 - Poker.py와 호환되도록 수정"""
        try:
            # 경로 디렉토리 생성
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Poker.py에서 읽을 수 있는 형태로 저장
            torch.save({
                'q_net_state_dict': self.q_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'train_step': self.train_step,
                'input_dim': self.input_dim,
                'action_num': self.action_num,
                # Poker.py에서 사용하는 키 추가
                'model_state_dict': self.q_net.state_dict(),
                'model_config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': 256,
                    'output_dim': self.action_num
                }
            }, path)
            print(f"✅ 모델 저장 완료: {path}")
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
    
    def load_model(self, path):
        """모델 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']
            print(f"✅ 모델 로드 완료: {path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

def normalize_observation(obs_data, target_dim):
    """관찰 데이터를 target_dim 크기로 정규화"""
    try:
        if obs_data is None:
            return np.zeros(target_dim, dtype=np.float32)
        
        # numpy 배열로 변환
        if isinstance(obs_data, (list, tuple)):
            obs_array = np.array(obs_data, dtype=np.float32)
        elif isinstance(obs_data, np.ndarray):
            obs_array = obs_data.astype(np.float32)
        elif isinstance(obs_data, (int, float)):
            obs_array = np.array([float(obs_data)], dtype=np.float32)
        else:
            return np.zeros(target_dim, dtype=np.float32)
        
        # 1차원으로 평탄화
        obs_array = obs_array.flatten()
        
        # 크기 조정
        if len(obs_array) >= target_dim:
            return obs_array[:target_dim]
        else:
            # 0으로 패딩
            result = np.zeros(target_dim, dtype=np.float32)
            result[:len(obs_array)] = obs_array
            return result
            
    except Exception as e:
        print(f"관찰 정규화 오류: {e}")
        return np.zeros(target_dim, dtype=np.float32)

def train_poker_agent(num_episodes=1000):
    """포커 에이전트 훈련"""
    print("🎰 텍사스 홀덤 에이전트 훈련 시작!")
    
    try:
        # RLCard 환경 생성
        env = make('no-limit-holdem', config={'seed': 42})
        print(f"✅ 환경 생성 완료: {env.game.get_num_players()}명 플레이어")
        
        # 환경 정보 출력 및 상태 크기 확인
        print(f"원본 상태 크기: {env.state_shape}")
        print(f"액션 수: {env.num_actions}")
        print(f"플레이어 수: {env.num_players}")
        
        # 실제 상태 차원 확인을 위한 테스트 실행
        print("🔍 실제 상태 차원 확인 중...")
        test_state, _ = env.reset()
        actual_obs = test_state['obs'] if isinstance(test_state, dict) else test_state
        
        if isinstance(actual_obs, (list, np.ndarray)):
            actual_obs = np.array(actual_obs).flatten()
            actual_input_dim = len(actual_obs)
        else:
            actual_input_dim = 1
            
        print(f"실제 관찰 차원: {actual_input_dim}")
        print(f"환경 state_shape: {env.state_shape}")
        
        # 실제 차원을 사용
        input_dim = actual_input_dim
        print(f"사용할 입력 차원: {input_dim}")
        
        # 에이전트 초기화
        agent = CustomDQNAgent(input_dim, env.num_actions)
        print(f"디바이스: {agent.device}")
        
        # 다른 플레이어들은 랜덤 에이전트로 설정
        agents = [agent]
        for i in range(env.num_players - 1):
            agents.append(RandomAgent(env.num_actions))
        
        env.set_agents(agents)
        
        # 훈련 기록
        episode_rewards = []
        losses = []
        win_rates = []
        
        for episode in range(num_episodes):
            try:
                # 게임 실행
                trajectories, payoffs = env.run(is_training=True)
                
                # 경험 저장
                if trajectories and len(trajectories) > 0:
                    agent_trajectory = trajectories[0]
                    
                    for i, ts in enumerate(agent_trajectory):
                        is_done = (i == len(agent_trajectory) - 1)
                        
                        if len(ts) >= 4:
                            state, action, reward, next_state = ts[:4]
                            
                            # 상태 처리
                            if isinstance(state, dict) and 'obs' in state:
                                obs = normalize_observation(state['obs'], input_dim)
                                modified_state = {
                                    'obs': obs, 
                                    'legal_actions': state.get('legal_actions', list(range(env.num_actions)))
                                }
                            else:
                                obs = normalize_observation(state, input_dim)
                                modified_state = {
                                    'obs': obs, 
                                    'legal_actions': list(range(env.num_actions))
                                }
                            
                            # next_state 처리
                            if isinstance(next_state, dict) and 'obs' in next_state:
                                next_obs = normalize_observation(next_state['obs'], input_dim)
                                modified_next_state = {
                                    'obs': next_obs, 
                                    'legal_actions': next_state.get('legal_actions', list(range(env.num_actions)))
                                }
                            else:
                                next_obs = normalize_observation(next_state, input_dim)
                                modified_next_state = {
                                    'obs': next_obs, 
                                    'legal_actions': list(range(env.num_actions))
                                }
                            
                            agent.feed((modified_state, action, reward, modified_next_state, is_done))
                
                # 에이전트 훈련
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train()
                    if loss > 0:
                        losses.append(loss)
                
                # 보상 기록
                reward = payoffs[0] if payoffs else 0
                episode_rewards.append(reward)
                
                # 승률 계산
                if len(episode_rewards) >= 100:
                    recent_rewards = episode_rewards[-100:]
                    wins = sum(1 for r in recent_rewards if r > 0)
                    win_rate = wins / len(recent_rewards)
                    win_rates.append(win_rate)
                
                # 진행상황 출력
                if episode % 100 == 0 and episode > 0:
                    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    current_win_rate = win_rates[-1] if win_rates else 0
                    print(f"에피소드 {episode:5d} | "
                          f"평균 보상: {avg_reward:6.2f} | "
                          f"승률: {current_win_rate:5.1%} | "
                          f"입실론: {agent.epsilon:.3f} | "
                          f"메모리: {len(agent.memory)}")
                
                # 모델 저장
                if episode % 500 == 0 and episode > 0:
                    agent.save_model(f'poker_model_{episode}.pth')
                    
            except Exception as e:
                print(f"에피소드 {episode}에서 오류: {e}")
                continue
        
        # 최종 모델 저장
        agent.save_model('final_poker_model.pth')
        
        return agent, episode_rewards, losses, win_rates
        
    except Exception as e:
        print(f"❌ 훈련 중 심각한 오류: {e}")
        import traceback
        traceback.print_exc()
        raise e

def evaluate_agent(agent, num_games=200):
    """훈련된 에이전트 평가"""
    print(f"\n🎯 에이전트 평가 ({num_games}게임)")
    
    try:
        # 평가용 환경
        env = make('no-limit-holdem', config={'seed': 123})
        
        # 다른 플레이어들은 랜덤
        agents = [agent]
        for i in range(env.num_players - 1):
            agents.append(RandomAgent(env.num_actions))
        env.set_agents(agents)
        
        payoffs = []
        wins = 0
        
        for game in range(num_games):
            try:
                # 평가 모드로 실행 (is_training=False)
                _, game_payoffs = env.run(is_training=False)
                payoff = game_payoffs[0]
                payoffs.append(payoff)
                
                if payoff > 0:
                    wins += 1
                
                if game % 50 == 0:
                    print(f"게임 {game:4d} 완료...")
                    
            except Exception as e:
                print(f"게임 {game}에서 오류: {e}")
                continue
        
        # 결과 분석
        if payoffs:
            avg_payoff = np.mean(payoffs)
            win_rate = wins / len(payoffs)
            std_payoff = np.std(payoffs)
            
            print(f"\n📊 평가 결과:")
            print(f"완료된 게임: {len(payoffs)}")
            print(f"승리: {wins}")
            print(f"승률: {win_rate:.1%}")
            print(f"평균 보상: {avg_payoff:.3f}")
            print(f"보상 표준편차: {std_payoff:.3f}")
        else:
            avg_payoff, win_rate = 0, 0
            print("❌ 평가할 수 있는 게임이 없습니다.")
        
        return avg_payoff, win_rate, payoffs
        
    except Exception as e:
        print(f"❌ 평가 중 오류: {e}")
        return 0, 0, []

def plot_training_results(rewards, losses, win_rates):
    """훈련 결과 시각화"""
    try:
        plt.figure(figsize=(15, 10))
        
        # 2x2 서브플롯
        plt.subplot(2, 2, 1)
        if rewards:
            plt.plot(rewards, alpha=0.3, color='blue', label='원본')
            if len(rewards) >= 50:
                smoothed = [np.mean(rewards[max(0, i-49):i+1]) for i in range(len(rewards))]
                plt.plot(smoothed, color='red', linewidth=2, label='평균(50)')
            plt.title('에피소드별 보상')
            plt.xlabel('에피소드')
            plt.ylabel('보상')
            plt.legend()
            plt.grid(True)
        
        plt.subplot(2, 2, 2)
        if losses:
            plt.plot(losses, color='orange')
            plt.title('훈련 손실')
            plt.xlabel('훈련 스텝')
            plt.ylabel('손실')
            plt.grid(True)
        
        plt.subplot(2, 2, 3)
        if win_rates:
            plt.plot(win_rates, color='green', linewidth=2)
            plt.title('승률 (100게임 단위)')
            plt.xlabel('에피소드 (x100)')
            plt.ylabel('승률')
            plt.ylim(0, 1)
            plt.grid(True)
        
        plt.subplot(2, 2, 4)
        if rewards:
            plt.hist(rewards, bins=30, alpha=0.7, color='purple')
            plt.title('보상 분포')
            plt.xlabel('보상')
            plt.ylabel('빈도')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"시각화 중 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🎲 RLCard를 사용한 텍사스 홀덤 강화학습")
    print("=" * 50)
    
    try:
        # 훈련 (테스트용으로 적은 에피소드)
        print("1️⃣ 에이전트 훈련 중...")
        agent, rewards, losses, win_rates = train_poker_agent(num_episodes=500)
        
        # 평가
        print("\n2️⃣ 에이전트 평가 중...")
        avg_payoff, win_rate, eval_payoffs = evaluate_agent(agent, num_games=100)
        
        # 결과 시각화
        print("\n3️⃣ 결과 시각화...")
        plot_training_results(rewards, losses, win_rates)
        
        print(f"\n🎊 훈련 완료!")
        print(f"최종 성능: 승률 {win_rate:.1%}, 평균 보상 {avg_payoff:.3f}")
        
        return agent
        
    except Exception as e:
        print(f"❌ 메인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trained_agent = main()