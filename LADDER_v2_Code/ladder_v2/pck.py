"""
Pedagogical Content Knowledge (PCK) base for LADDER v2.

Shulman (1986): PCK is "how to teach this specific content" — the knowledge
that distinguishes a subject-matter expert from a subject-matter teacher.

Currently covers 5 core linear algebra concepts. Each entry maps to the
SHaPE benchmark KP vocabulary.
"""

# Maps concept name → {common_misconceptions, counter_examples, entry_points}
PCK_BASE = {
    "Linear Dependence and Independence": {
        "common_misconceptions": [
            "以为线性相关就是向量成比例（这只对两个向量成立，三个以上不适用）",
            "混淆'线性相关'和'线性组合'——能线性表示不代表相关",
            "以为线性无关的向量必须正交",
        ],
        "counter_examples": [
            "构造 (1,0,0), (2,0,0), (0,1,0) —— 前两个相关但第三个打破了比例关系",
            "三个共面向量在三维空间中线性相关（即使不成比例）",
        ],
        "entry_points": {
            "memory": "从'齐次方程组是否有非零解'切入，给一个2×2的例子",
            "comprehension": "从几何直观切入：两个向量共线=相关，三个向量共面=相关",
            "application": "给一个3×3矩阵，判断列向量是否线性独立（联系行列式）",
        },
    },

    "The Rank of a Matrix": {
        "common_misconceptions": [
            "混淆'秩'和'维数'——矩阵的秩是其列空间的维数，不是矩阵的尺寸",
            "以为满秩矩阵一定是可逆的（只对方阵成立）",
            "以为秩不能超过行数或列数的最小值是最优估计（其实这是秩的定义性上界）",
        ],
        "counter_examples": [
            "3×2矩阵的秩最多为2（非方阵也有秩），打破'秩=min(行,列)且只有方阵有秩'的误解",
            "一个3×3秩为2的矩阵——列向量共面但不共线",
        ],
        "entry_points": {
            "memory": "从'矩阵化成行阶梯形后非零行的数量'切入",
            "comprehension": "从'列空间的维数'切入——秩=独立列向量的个数",
            "application": "给一个实际数据矩阵（如评分矩阵），问'这个系统有多少独立因素'",
        },
    },

    "Subspaces of N-Dimensional Space": {
        "common_misconceptions": [
            "以为子空间必须过原点（这个是对的）但以为任何平面都是R3的子空间（必须过原点）",
            "混淆子空间和子集——子空间对加法和数乘封闭",
            "以为零空间={0}和零空间为空集是一回事",
        ],
        "counter_examples": [
            "平面 x+y+z=1 不是R3的子空间（不过原点，不满足零向量条件）",
            "两个不共线的向量张成一个过原点的平面（是子空间）",
        ],
        "entry_points": {
            "memory": "从'子空间三条件：含零向量、加法封闭、数乘封闭'切入",
            "comprehension": "从几何直观：R3中过原点的线和面是子空间",
            "application": "判断一个给定的向量集合是否构成子空间",
        },
    },

    "Eigenvalues and Eigenvectors": {
        "common_misconceptions": [
            "以为每个矩阵都有实特征值（非对称矩阵可能有复特征值）",
            "以为特征向量唯一（任何非零倍数都是特征向量）",
            "混淆特征值和矩阵对角元素",
        ],
        "counter_examples": [
            "旋转矩阵 [[0,-1],[1,0]] 没有实特征值（特征值为±i）",
            "如果v是特征向量，2v也是——说明特征向量不唯一",
        ],
        "entry_points": {
            "memory": "从'Av=λv, v≠0'的定义方程切入",
            "comprehension": "从几何意义切入：特征向量是被矩阵变换后方向不变的向量",
            "application": "给一个2×2矩阵，手算特征值和特征向量",
        },
    },

    "The Determinant of an NxN Matrix": {
        "common_misconceptions": [
            "以为det(A+B)=det(A)+det(B)（行列式不满足加法）",
            "以为det(A)=0意味着矩阵没有逆（这个对）但以为det(A)≠0一定可逆（只对方阵）",
            "混淆行列式和矩阵的'大小'——行列式是缩放因子，可正可负",
        ],
        "counter_examples": [
            "A=[[1,0],[0,1]], B=[[-1,0],[0,-1]]: det(A+B)=det([[0,0],[0,0]])=0, 但det(A)+det(B)=1+1=2",
            "行列式为负：[[0,1],[1,0]]的行列式=-1，表示变换翻转了定向",
        ],
        "entry_points": {
            "memory": "从2×2公式 ad-bc 切入",
            "comprehension": "从几何意义切入：行列式=平行四边形的有向面积",
            "application": "判断一个线性方程组是否有唯一解（Cramer's rule）",
        },
    },
}


def get_pck(concept_name: str) -> dict:
    """Retrieve PCK entry for a given concept. Returns empty dict if not found."""
    for key, entry in PCK_BASE.items():
        if key.lower() == concept_name.lower():
            return entry
    # Fuzzy match
    for key, entry in PCK_BASE.items():
        if concept_name.lower() in key.lower() or key.lower() in concept_name.lower():
            return entry
    return {}


def get_misconception_hint(concept_name: str) -> str:
    """Generate a misconception-aware hint for Agent B prompt injection."""
    pck = get_pck(concept_name)
    if not pck:
        return ""

    misconceptions = pck.get("common_misconceptions", [])
    counter_examples = pck.get("counter_examples", [])

    parts = []
    if misconceptions:
        parts.append(f"学生可能对'{concept_name}'有以下常见误解：")
        for i, m in enumerate(misconceptions[:2]):  # Top 2 most common
            parts.append(f"  {i+1}. {m}")
    if counter_examples:
        parts.append(f"有效的反例：{counter_examples[0]}")

    return "\n".join(parts)


def get_entry_point(concept_name: str, bloom_level: str) -> str:
    """Get the best entry point for teaching a concept at a given Bloom level."""
    pck = get_pck(concept_name)
    if not pck:
        return ""
    entry_points = pck.get("entry_points", {})
    # Fallback chain: exact match → comprehension → application → first available
    return (entry_points.get(bloom_level)
            or entry_points.get("comprehension")
            or next(iter(entry_points.values()), ""))


# ===========================================================================
if __name__ == "__main__":
    for name in ["Linear Dependence and Independence", "The Rank of a Matrix",
                 "Eigenvalues and Eigenvectors", "Nonexistent Concept"]:
        pck = get_pck(name)
        print(f"\n{'='*60}")
        print(f"Concept: {name}")
        print(f"Found: {bool(pck)}")
        if pck:
            print(f"Misconceptions: {len(pck.get('common_misconceptions',[]))}")
            print(f"Counter-examples: {len(pck.get('counter_examples',[]))}")
            hint = get_misconception_hint(name)
            print(f"Hint preview: {hint[:120]}...")
            for level in ["memory", "comprehension", "application"]:
                ep = get_entry_point(name, level)
                print(f"  Entry ({level}): {ep[:80]}...")
    print("\nDONE")
