import random
from typing import List, Tuple
from writer import save_liveness_to_json

def processa_move_related(num_variables: int, num_moves: int) -> Tuple[List[int], List[int]]:
    """
    Gera a lista de índices de variáveis que serão pareadas como moves.
    As variáveis de move são os pares (a, b) tais que o tempo de vida de "a" termina
    no começo de "b", e tem a forma:

    b = a

    Esse é um caso que o algoritmo de George-Appel otimiza e precisa ser informado ao algoritmo

    Para deixar mais aleatório quais são selecionadas, o shuffle é utilizado.
    """

    all_indices = list(range(num_variables))

    random.shuffle(all_indices)

    move_related_indices = all_indices[:num_moves * 2]
    normal_indices = all_indices[num_moves * 2:]

    return (normal_indices, move_related_indices)
        
def generate_liveness_with_moves(
    id: int,
    num_variables: int = 50,
    num_moves: int = 5,
    inferior_limit: int = 2,
    superior_limit: int = 15,
    code_lines: int = 500
):
    """
    Gera uma lista de locais de uso para uma configuração especificada
    de variáveis. Simula também a criação de moves entre as variáveis.

    Argumentos:
        id: Identificador para salvar o arquivo.
        num_variables: O total de variáveis geradas.
        num_moves: A quantidade de moves entre as variáveis.
        inferior_limit: A quantidade mínima de usos de cada variável.
        superior_limit: A quantidade máxima de usos de cada variável.
        code_lines: O número total de linhas de código.

    Retorno:
        Salva o estado inicial em um arquivo JSON.
    """

    # --- Validação de entrada ---
    if num_variables < num_moves * 2:
        raise ValueError("Número de variáveis insuficientes para criar o número de moves desejados")
    if inferior_limit < 2:
        raise ValueError("Limite inferior deve ser maior ou igual a 2")
    if superior_limit < inferior_limit:
        raise ValueError("Limite superior deve ser maior que limite inferior")

    # --- 0. Instanciação das listas ---
    liveness: List[List[int]] = [[] for _ in range(num_variables)]
    moves_generated: List[Tuple[int, int]] = []

    # --- 1. Seleciona variáveis para o move ---
    normal_indices, move_related_indices = processa_move_related(num_variables, num_moves)

    # --- 2. Gera pares de move ---
    for i in range(0, len(move_related_indices), 2):
        # var1 é o destino, var2 é a origem (para o move var1 = var2)
        var1_dest = move_related_indices[i]
        var2_source = move_related_indices[i+1]
        moves_generated.append((var1_dest, var2_source))

        # Pega mais ou menos entre 1/5 das linhas e 4/5 das linhas
        move_line = random.randint(code_lines // 5, code_lines * 4 // 5)

        source_sites = [move_line]
        num_source_usages = random.randint(inferior_limit - 1, superior_limit - 1)
        for _ in range(num_source_usages):
            source_sites.append(random.randint(1, move_line - 1))
        liveness[var2_source] = source_sites

        dest_sites = [move_line]
        num_dest_usages = random.randint(inferior_limit - 1, superior_limit - 1)
        for _ in range(num_dest_usages):
            dest_sites.append(random.randint(move_line + 1, code_lines))
        liveness[var1_dest] = dest_sites

    # --- 3. Gera o liveness para as variáveis restantes ---
    for var_index in normal_indices:
        num_usages = random.randint(inferior_limit, superior_limit)
        liveness[var_index] = [random.randint(1, code_lines) for _ in range(num_usages)]

    # --- 4. Ordena e remove duplicatas ---
    for i in range(num_variables):
        if liveness[i]:
            liveness[i] = sorted(list(set(liveness[i])))

    result = dict()

    result["usage_sites"] = liveness
    result["moves"] = moves_generated

    save_liveness_to_json(result, f"data/liveness-{id}.json")


if __name__ == '__main__':
    generate_liveness_with_moves(id=1)
    generate_liveness_with_moves(id=2, num_variables=50, num_moves=2, code_lines=500)
    generate_liveness_with_moves(id=3, num_variables=1000, num_moves=100, code_lines=10000)
    generate_liveness_with_moves(id=4, num_variables=10000, num_moves=3000, code_lines=20000)
    generate_liveness_with_moves(id=5, num_variables=10000, num_moves=100, code_lines=100000)
    

