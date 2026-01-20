# 키워드 맵
keyword_map = {
    "양자 얽힘": "entanglement", "양자얽힘": "entanglement", "얽힌 상태": "entanglement", "얽힘": "entanglement",
    "얽힌상태": "entanglement", "quantum entanglement": "entanglement", "entangled state": "entanglement",
    "벨 상태": "entanglement", "벨상태": "entanglement", "Bell 상태": "entanglement", "Bell state": "entanglement",
    "bell state": "entanglement", "bellstate": "entanglement", "EPR": "entanglement", "EPR 역설": "entanglement", 
    "EPR paradox": "entanglement", "CHSH": "entanglement", "CHSH inequality": "entanglement", "벨 부등식": "entanglement",

    "양자 중첩": "superposition", "양자중첩": "superposition",
    "quantum superposition": "superposition", "superposition": "superposition",
    "중첩 상태": "superposition", "중첩상태": "superposition", "중첩": "superposition",
    "슈뢰딩거의 고양이": "superposition", "슈뢰딩거 고양이": "superposition",
    "슈뢰딩거의고양이": "superposition", "슈뢰딩거고양이": "superposition", "Schrodinger's cat": "superposition",

    "양자 터널링": "tunneling", "양자터널링": "tunneling", "quantum tunneling": "tunneling","터널링": "tunneling",
    "tunneling": "tunneling", "터널 효과": "tunneling", "터널링 효과": "tunneling",

    "불확정성 원리": "uncertainty", "불확정성원리": "uncertainty", "uncertainty principle": "uncertainty", "하이젠베르크 원리": "uncertainty",
    "불확정성의 원리": "uncertainty", "불확정성": "uncertainty",

    "양자 컴퓨터": "quantum_computer", "양자컴퓨터": "quantum_computer",
    "quantum computer": "quantum_computer", "quantum computing": "quantum_computer",
    "큐비트": "quantum_computer", "큐빗": "quantum_computer",
    "양자 비트": "quantum_computer", "quantum bit": "quantum_computer", "qubit": "quantum_computer",
    "양자 게이트": "quantum_computer", "양자게이트": "quantum_computer", "quantum gate": "quantum_computer",
    "양자 연산 게이트": "quantum_computer", "양자 연산": "quantum_computer", "양자연산": "quantum_computer",
    "양자 상태": "quantum_computer", "양자상태": "quantum_computer", "quantum state": "quantum_computer", "quantumstate": "quantum_computer",
    "큐비트 상태": "quantum_computer", "큐비트상태": "quantum_computer", "qubit state": "quantum_computer", "qubit 상태": "quantum_computer",
    "양자 회로": "quantum_computer", "양자회로": "quantum_computer", "quantum circuit": "quantum_computer", "quantum_circuit": "quantum_computer","회로": "quantum_computer",
    "큐비트 회로": "quantum_computer", "큐비트회로": "quantum_computer", "양자 논리 회로": "quantum_computer", "quantum logic circuit": "quantum_computer",

    "블로흐 구": "bloch_sphere", "블로흐구": "bloch_sphere", "Bloch 구": "bloch_sphere", "Bloch sphere": "bloch_sphere","블로흐": "bloch_sphere",
    "bloch sphere": "bloch_sphere", "blochsphere": "bloch_sphere", "블로흐 스피어": "bloch_sphere",
    "양자 상태 구": "bloch_sphere", "큐비트 구": "bloch_sphere", "큐비트 상태 구": "bloch_sphere", "큐비트 시각화": "bloch_sphere",

    "입자성": "wave_particle_duality","입자": "wave_particle_duality","파동": "wave_particle_duality","파동성": "wave_particle_duality",
    "입자성과 파동성": "wave_particle_duality","입자성과파동성": "wave_particle_duality","파동성과 입자성": "wave_particle_duality","파동성과입자성": "wave_particle_duality",
    "이중슬릿": "wave_particle_duality","이중 슬릿": "wave_particle_duality","이중슬릿 실험": "wave_particle_duality","이중 슬릿 실험": "wave_particle_duality",
    "wave": "wave_particle_duality","particle": "wave_particle_duality","wave particle": "wave_particle_duality","wave-particle": "wave_particle_duality",
    "particle wave": "wave_particle_duality","particle-wave": "wave_particle_duality","wave particle duality": "wave_particle_duality","wave-particle duality": "wave_particle_duality","particle wave duality": "wave_particle_duality","particle-wave duality": "wave_particle_duality","double slit": "wave_particle_duality","double-slits": "wave_particle_duality","double slit experiment": "wave_particle_duality",
    "double-slits experiment": "wave_particle_duality","doubleslit": "wave_particle_duality","two slit experiment": "wave_particle_duality","two-slits experiment": "wave_particle_duality",

    "양자 텔레포테이션": "quantum_teleportation", "텔레포테이션": "quantum_teleportation", "양자텔레포테이션": "quantum_teleportation", "큐비트 전송": "quantum_teleportation", "양자 정보 전달": "quantum_teleportation","quantum_teleportation":"quantum_teleportation",
    "quantum teleportation": "quantum_teleportation", "quantumteleportation": "quantum_teleportation","양자 순간 이동": "quantum_teleportation", "양자 텔포": "quantum_teleportation", "텔포": "quantum_teleportation"

    }

# 개념 설명
concepts = {
"entanglement": {
    "개념 설명": """\
        양자 얽힘은 두 입자가 서로 얽혀 있어서 한 입자의 상태를 측정하는 순간 다른 입자의 상태도 즉시 결정되는 현상입니다.
        이 현상은 두 입자가 아무리 멀리 떨어져 있어도 유지됩니다.
        대표적인 예로 두 큐비트가 최대로 얽힌 벨 상태(Bell states)가 있습니다.""",

    "예시": """\
        예를 들어, 얽힌 두 입자 A와 B가 있다고 해봅시다.
        A를 서울에 두고 B를 뉴욕으로 보냈다고 가정합니다.
        만약 서울에서 A의 스핀을 측정해서 위쪽(up)으로 나왔다면,
        같은 순간 뉴욕에 있는 B의 스핀은 아래쪽(down)으로 즉시 결정됩니다.
        
        → 이런 방식으로 얽힌 입자들은 거리에 관계없이 정보를 공유하는 것처럼 보입니다.

        [벨 상태 예시]
        가장 대표적인 벨 상태는 다음과 같습니다.
        > (|00⟩ + |11⟩)/√2 

        이 상태에서는 두 큐비트가 모두 0이거나 모두 1인 상태로 얽혀 있으며,
        한 큐비트가 측정되어 0이 나오면 다른 큐비트도 0으로 확정됩니다.
        """,

    "특징 정리": """\
        - 즉각적 상관관계
        - 거리와 무관한 연결성
        - 대표 예: 벨 상태""",

    "관련 개념": ["양자 중첩", "양자 텔레포테이션"]
    },


    "superposition": {
        "개념 설명": """\
        양자 중첩은 하나의 입자가 동시에 여러 상태를 가질 수 있다는 양자역학의 핵심 개념입니다.
        고전역학에서는 물체가 하나의 상태에만 존재할 수 있지만, 양자역학에서는 그렇지 않습니다.

        이를 설명할 때 자주 등장하는 예가 슈뢰딩거의 고양이 사고실험입니다.
        밀폐된 상자 안에서 고양이는 관측 전까지 '살아 있음'과 '죽어 있음'이 동시에 중첩된 상태로 존재한다고 볼 수 있습니다.
        """,

        "예시": """\
        전자의 스핀 상태가 위(up)와 아래(down) 중 하나로 존재한다고 생각해봅시다.  
        고전적으로는 둘 중 하나를 선택해야 하지만 
        양자역학에서는 전자가 측정되기 전까지 위와 아래 두 상태가 동시에 존재하는 중첩 상태에 있을 수 있습니다.
        
        슈뢰딩거의 고양이 실험도 같은 원리로 이해할 수 있습니다. 
        상자를 열어 관측하는 순간, 중첩 상태는 하나로 확정됩니다.

        → 이러한 성질은 양자 컴퓨터의 큐비트처럼 병렬 계산이 가능한 기반이 됩니다.""",

        "특징 정리": """\
        - 여러 상태의 동시 존재
        - 관측 순간 하나로 붕괴
        - 대표 실험: 슈뢰딩거의 고양이""",

        "관련 개념": "양자 얽힘"
        },  
    
    "uncertainty": {
        "개념 설명": """\
        불확정성 원리는 양자역학의 기본 개념으로, 입자의 위치와 운동량을 동시에 정확하게 알 수 없다는 뜻입니다.  
        위치를 더 정확히 알수록 운동량은 불확실해지고, 
        운동량을 더 정밀하게 알수록 위치는 더 불확실해집니다.

        수학적으로는 하이젠베르크의 관계식 Δx·Δp ≥ ħ/2 로 표현됩니다.
        """,

        "예시": """\
        전자의 위치를 현미경으로 매우 정확하게 관찰한다고 해봅시다.
        그러면 전자가 어디 있는지는 잘 알 수 있지만,
        얼마나 빠르게 움직이는지는 애매해집니다.

        반대로 전자의 운동량을 정밀하게 알면,
        전자가 어디에 있는지는 흐릿하게 퍼져 있는 것처럼만 알 수 있습니다.
        """,

        "특징 정리": """\
        - 위치·운동량 동시 정밀 측정 불가
        - Δx·Δp ≥ ħ/2 (하이젠베르크 관계식)
        - 파동함수의 푸리에 변환에서 기인
        - 고전 물리와의 큰 차이
        """
    },


    "tunneling": {
            "개념 설명": """\
            양자 터널링은 입자가 고전적으로는 넘을 수 없는 장벽을 
            확률적으로 통과하는 양자역학적인 현상입니다.  
            에너지가 부족해도 입자가 장벽을 뚫고 반대편으로 이동할 수 있다는 점이 특징입니다.
            """,

            "예시": """\
            공이 장벽을 넘으려면 장벽보다 큰 에너지가 필요합니다.
            하지만 공이 장벽을 마주했을 때 그보다 낮은 에너지를 가졌더라도  
            작은 확률로 장벽을 뚫고 반대편으로 나오는 현상을 말합니다.

            → 이는 전자가 입자이면서도 파동의 성질을 가지기 때문에
            그 파동이 장벽을 일부 통과하면서 반대쪽에서 다시 나타날 수 있기 때문입니다.""",

            "특징 정리": """\
            - 낮은 에너지로도 장벽 통과 가능
            - 파동 성질로 생기는 확률적 현상
            - 고전 물리에서는 불가능
            - 핵융합·전자소자 등에 응용
            """,
    },

    "bloch_sphere": {
        "개념 설명": """\
        블로흐 구는 큐비트의 상태를 구로 표현한 그림입니다.  
        큐비트는 0과 1이 섞인 상태인 중첩 상태일 수 있습니다.
	    이를 구 위의 한 점으로 나타내면 훨씬 이해하기 쉬워집니다.
    """,
        "예시": """\
        - 큐비트가 |0⟩ 상태라면 구의 위쪽(북극)에 있습니다.
        - |1⟩ 상태라면 구의 아래쪽(남극)에 있습니다.
        - 0과 1이 반반 섞인 상태라면 구의 적도에 위치합니다.
        - 양자 게이트(X, H, Z 등)는 이 점을 구 위에서 회전시키는 동작으로 표현됩니다.
        
        → 블로흐 구는 큐비트가 어떤 상태인지 눈으로 직관적으로 보여줍니다.""",

        "특징 정리": """\
        - 큐비트 상태를 구 위의 점으로 표현
        - 수학 대신 그림으로 직관 제공
        - 게이트 작용을 회전으로 시각화
        - 양자 컴퓨터 학습의 핵심 도구
        """,

        "관련 개념":"양자 컴퓨터"
    },

    "quantum_computer": {
        "개념 설명": """\
        양자 컴퓨터는 양자역학의 원리를 이용해 정보를 처리하는 계산 장치입니다.
        고전 컴퓨터가 0과 1의 비트를 사용하는 반면, 양자 컴퓨터는 큐비트를 사용합니다.
        큐비트는 양자 상태를 가질 수 있어 동시에 0과 1 상태를 함께 표현할 수 있습니다.

        양자 상태는 큐비트의 모든 가능한 상태를 수학적으로 나타낸 것으로,
        중첩과 간섭, 얽힘을 통해 병렬적인 계산이 가능하게 합니다.

        양자 컴퓨터는 양자 회로를 통해 동작합니다.
        회로 안에서는 양자 게이트가 큐비트 상태를 바꾸거나 서로 얽히게 만듭니다.
        게이트는 고전 컴퓨터의 논리 게이트와 비슷한 역할을 하지만 계산 방식은 양자 법칙을 따릅니다.

        이러한 특성 덕분에 양자 컴퓨터는 고전 컴퓨터에 비해 특정 문제를 훨씬 더 빠르게 풀 수 있습니다.
        """,

        "예시": """\
        - 고전 컴퓨터: 3개의 비트가 '101'과 같이 하나의 상태만 가짐  
        - 양자 컴퓨터: 3개의 큐비트가 '000'부터 '111'까지 8가지 상태를 동시에 표현 가능
        - 양자 게이트: X, H, CNOT 게이트 등을 이용해 큐비트 상태를 회전·변화시킴  
        """,

        "특징 정리": """\
        - 큐비트: 0과 1이 겹친 중첩 상태 가능
        - 양자 게이트: 큐비트를 회전·얽히게 만드는 연산
        - 양자 회로: 게이트들을 연결해 계산 수행
        - 병렬 계산: 중첩·간섭·얽힘으로 동시에 많은 경우 처리
        """,
    },

    "wave_particle_duality": {
        "개념 설명": """\
        입자성과 파동성은 빛과 물질이 동시에 입자의 성질과 파동의 성질을 모두 가진다는 양자역학의 핵심 개념입니다.
        고전 물리학에서는 빛은 파동, 전자 같은 물질은 입자로만 생각했지만, 
        실험 결과 두 성질이 함께 나타난다는 것이 확인되었습니다.

        이 현상을 가장 잘 보여주는 것이 이중 슬릿 실험입니다.
        두 개의 좁은 슬릿을 통과한 빛이나 전자는 스크린에 간섭 무늬를 만듭니다.
        이 간섭 무늬는 파동성의 증거이며 동시에 입자를 하나씩 쏘아도 시간이 지나면 동일한 무늬가 나타납니다.
        이는 입자 하나가 스스로 간섭하는 것처럼 보입니다.

        하지만 슬릿을 통과하는 경로를 관측하면, 간섭 무늬가 사라지고 단순히 두 개의 줄만 나타납니다.
        즉, 관측이 이루어지는 순간 파동성이 사라지고 입자성만 드러납니다.
        """,

        "예시": """\
        - 빛: 프리즘을 통과하면 굴절·간섭(파동성), 광전효과에서는 광자(입자성)로 행동  
        - 전자: 이중 슬릿 실험에서 간섭 무늬 형성(파동성), 금속 표면 충돌 시 개별 입자처럼 반응(입자성)  
        """,

        "특징 정리": """\
        - 빛·물질 모두 입자와 파동 성질
        - 파동성: 간섭·회절
        - 입자성: 개별 알갱이
        - 관측 시 파동성 소멸""",

    },

    "quantum_teleportation": {
        "개념 설명": """\
        양자 텔레포테이션은 입자 자체를 보내는 것이 아니라
        입자의 양자 상태를 다른 곳에 옮기는 기술입니다.

        이를 위해 두 사람이 미리 공유한 얽힌 입자 쌍과 두 비트의 고전 정보를 사용합니다.
        앨리스가 자신의 입자를 측정하면 원래 상태는 사라지고,
        그 결과를 밥에게 알려주면 밥은 받은 정보로
        자신의 입자를 조작해 원래 상태를 되살릴 수 있습니다.

        이 과정에서 물체가 순간 이동하는 것은 아니고
        오직 양자 상태만 옮겨집니다.
        """,

        "예시": """\
        앨리스가 미지의 상태 |ψ⟩를 밥에게 보내려고 한다.
        두 사람은 미리 얽힌 쌍을 나눠 가진다.
        앨리스가 자신의 큐비트를 측정하고 결과를 밥에게 보낸다.
        밥은 그 결과에 맞춰 연산을 적용해 |ψ⟩를 얻는다.

        → 이 과정에서 원본은 사라지고 밥의 큐비트에 상태가 재현된다.
        """,

        "특징 정리": """\
        - 상태 전송 O 입자 이동 X
        - 얽힌 쌍과 고전 2비트 필요
        - 원본 상태는 측정 시 사라짐
        """,

        "관련 개념": "양자 얽힘"
    }

}

from textwrap import dedent

SIM_REGISTRY = {
    "entanglement": [
        ("quantum_entanglement_simulation", "양자 얽힘 시뮬레이터",
        """\
        이 시뮬레이터는 얽힌 두 큐비트 중 하나를 측정했을 때,
        다른 큐비트의 상태가 즉시 결정되는 현상을 보여줍니다.
        - 왼쪽 그래프: 특정 방향에서 측정했을 때 결과 확률 분포.
        - 오른쪽 블로흐 구: 측정 이후 남은 큐비트의 상태 벡터가 자동 회전하며 시각화됨.
        - 특징: 한 큐비트를 측정하면 나머지 큐비트의 상태가 확률적으로 결정되는,
        양자 얽힘의 순간적 상관관계를 확인할 수 있음.
        """),
        ("bell_state", "벨 상태 시뮬레이터",
        """\
        이 시뮬레이터는 대표적인 얽힘 상태인 벨 상태를 보여줍니다.
        - 혼자 보면 각 입자의 결과는 0과 1이 50%인 무작위지만,
        같은 방향으로 두 입자를 같이 측정하면 결과가 항상 서로 같습니다.
        - 히트맵: 00, 11 칸만 밝게 보임 → 둘의 결과가 일치함.
        - 그래프: 두 측정 방향의 각도 차(Δθ) 에 따라 일치 정도가 변함.
        0° → 완전히 같음(+1), 90° → 무관(0), 180° → 완전히 반대(−1)
        (참고: 같을 확률 ≈ (1 + cos Δθ)/2)
        """)
    ],

    "wave_particle_duality": [
        ("double_slit", "이중 슬릿 간섭 시뮬레이터",
        """\
        이 시뮬레이터는 **빛의 입자성과 파동성**을 동시에 보여줍니다.  
        - 위 그래프: 조건에 따른 간섭 무늬가 보임.
        - 아래 그래프: ‘Emit’으로 입자를 하나씩 쏘면 점이 누적되어, 결국 위와 같은 무늬가 드러남.
        - 슬라이더: 파장 λ(nm), 슬릿 폭 a(μm), 슬릿 간격 d(μm), 스크린 거리 L(m), 위상 일치도(γ).
        - Mode: Single(회절무늬) / Double(간섭무늬)
        - W-Path: 어느 슬릿을 지났는지 알게 되면 위상 일치도 γ를 잃어 간섭무늬가 사라짐.
        """)
    ],

    "quantum_computer": [
        ("quantum_measure", "양자 측정 시뮬레이터",
        """\
        이 시뮬레이터는 큐비트의 초기 상태와 측정 후 상태를 블로흐 구에서 시각화해 보여줍니다.
        - 블로흐 구: 큐비트의 상태를 3차원 화살표로 표시
        - 버튼: show_initial → 초기 상태 표시, show_measurement → 측정 후 상태 표시
        - 측정 과정: 초기 상태 벡터가 측정 축에 수직 투영되어 |0⟩ 또는 |1⟩로 확정됨
        """),
        ("quantum_circuit_demo", "양자 회로 시뮬레이터",
        """\
        이 시뮬레이터는 큐비트 양자 회로를 구성하고 동작을 확인할 수 있습니다.
        - 큐비트 라인: q0, q1에 게이트를 적용
        - State vector: 현재 양자 상태를 벡터로 표시
        - 확률 그래프: |00⟩, |01⟩, |10⟩, |11⟩ 측정 확률을 시각화
        - 게이트 버튼: H, X, Y, Z, S, T 및 CNOT
        - 회전 슬라이더: Rx, Ry, Rz 회전 연산 적용
        - 큐비트 선택 옵션:
        1 qubit / 2 qubits: 회로에 사용할 큐비트 개수 설정
        target q0/q1: 단일 게이트 적용 대상 지정
        ctrl q0/q1 + tgt q0/q1: CNOT 게이트에서 제어(ctrl)와 대상(tgt) 큐비트 선택
        """),
        ("qubit_sim_hs", "큐비트 측정 시뮬레이터",
        """\
        이 시뮬레이터는 단일 큐비트를 임의의 상태로 준비한 뒤,
        측정을 통해 |0⟩ 또는 |1⟩ 결과가 확률적으로 나오는 과정을 보여줍니다.
        - 왼쪽 원판: 측정 결과를 룰렛처럼 시각화 (초록=|0⟩, 빨강=|1⟩).
        - 오른쪽 그래프: 여러 번 측정했을 때 |0⟩, |1⟩이 나온 비율을 확률로 표시.
        - 슬라이더: θ, φ로 상태 |ψ⟩ 조절
        - 옵션: Hadamard(H) 게이트 적용, 측정 1회/100회 실행
        """),
        ("qubit_simulator_renew", "큐비트 상태 시뮬레이터",
        """\
        이 시뮬레이터는 단일 큐비트의 상태를 블로흐 구 위에서 시각화하고,
        다양한 양자 게이트와 회전을 적용했을 때 상태가 어떻게 변하는지 보여줍니다.
        - 위 블로흐 구: 현재 큐비트 상태 벡터를 3차원 구 위에서 표시.
        - 아래 확률 그래프: |0⟩, |1⟩로 측정될 확률을 막대그래프로 표시.
        - 오른쪽 게이트 버튼: H, X, Y, Z, S, T 게이트를 적용해 상태를 변환.
        - 회전 슬라이더: θx, θy, θz 각도를 지정해 임의의 회전 연산(Rx, Ry, Rz)을 적용 가능.
        """),
        ("quantum_gate", "큐비트 게이트 시뮬레이터",
        """\
        이 시뮬레이터는 두 큐비트에 게이트를 적용하고 상태 변화와 측정 결과를 보여줍니다.
        - 블로흐 구: 큐비트 A, B의 부분 상태를 벡터로 표시
        - 확률 막대: |00>, |01>, |10>, |11> 측정 확률 분포
        - 엔트로피: S(ρ_A), S(ρ_B)로 얽힘 정도 확인 가능
        - 슬라이더: Rx, Ry, Rz 회전과 Global Phase 조절
        - 게이트 버튼: X, Y, Z, H, S, T, CNOT, CZ, SWAP
        - 측정 버튼: A, B 각각 Z축에서 측정
        """)
    ],

    "tunneling": [
        ("quantum_tunneling_simulation", "양자 터널링 시뮬레이터",
         """이 시뮬레이터는 2차원 퍼텐셜 장벽에서 파동의 터널링 현상을 보여줍니다.
- 입력값: 장벽 높이 V0, 장벽 너비 a, 초기 운동량 kx0
- 1D Slice: y=0에서 확률 밀도 |ψ(x,0)|²
- 2D Top View: 확률 밀도의 평면 히트맵
- 3D Surface: 확률 밀도의 입체 그래프
- 종료 시: 장벽 투과 확률과 터널링 성공 여부 출력""")
    ],

    "quantum_teleportation": [
        ("quantum_teleport_numpy", "양자 텔레포테이션 시뮬레이터",
         """이 시뮬레이터는 양자의 순간이동 과정을 단계별로 보여줍니다.
 - 입력 슬라이더: θ, φ 값 조정 → 보내고 싶은 상태 정함
 - Bell 측정: 앨리스가 자신의 큐비트를 측정해 두 개의 결과를 얻음
 - 고전 비트 전송: 앨리스는 그 결과를 밥에게 일반적인 신호로 보냄
 - 밥의 보정: 밥은 받은 신호에 따라 X, Z 같은 연산을 해 큐비트에 원래 상태를 되살림
 - 버튼: 측정, 비트 전송, 다시 시작""")
    ],

    "superposition": [
        ("wavefunction_focus_sim", "파동함수 시뮬레이터",
         """이 시뮬레이터는 입자의 파동함수가 시간에 따라 어떻게 변하는지 보여줍니다.
 - 그래프 1: 확률 분포 |ψ(x,t)|² + 퍼텐셜 V(x)  
 - 그래프 2: 파동의 위상 변화
 - 그래프 3: 확률 전류 j(x)

[버튼]  
 - 초기 상태: Single(단일 파동) / Two(이중 파동) / HO n=0,1(조화진동자 상태) 
 - 퍼텐셜: Free(자유) / Harmonic(조화) / Barrier(장벽)

[슬라이더]  
- dt(시간 스텝), CAP(경계 흡수), φ_g(표시용 위상), Potential param(ω 또는 장벽 높이)
- x(위치), k(운동량), σ(폭), Δφ(위상차)
- Barrier width(폭), center(중심)
 """),
        ("quantum_walk_sim_3d", "양자 보행 시뮬레이터",
         """이 시뮬레이터는 양자 보행과 고전적 보행의 차이를 시각화합니다.
 - 그래프 1: 최근 200 스텝 동안의 양자 확률 분포
 - 그래프 2: Quantum Walk 확률 분포 P_Q(x)
 - 그래프 3: Classical Walk 확률 분포 P_C(x)

[버튼]  
 - 경계 조건: absorb(흡수) / reflect(반사)
 - 모드: Quantum / Classical

[슬라이더]  
 - θ(각도), φ(초기 위상)
 - steps per frame(프레임당 스텝 수)
 - MAX steps(최대 스텝, 자동 종료)
 - 3D steps shown(3D 표시 스텝 범위)""")
    ],

    "uncertainty": [
        ("uncertainty", "불확정성 시뮬레이터",
         """이 시뮬레이터는 불확정성 원리를 보여줍니다.  
- 왼쪽 그래프: |ψ(x)|² 위치 분포  
- 오른쪽 그래프: |φ(k)|² 운동량 분포 

[슬라이더]  
- σ_x : 위치 폭 조절  
- x0 : 초기 중심 위치  
- k0 : 초기 운동량 
- L : 시뮬레이션 구간 길이  
- x_meas : 측정 중심 위치  
- intensity : 측정 강도  
""")
    ],
}