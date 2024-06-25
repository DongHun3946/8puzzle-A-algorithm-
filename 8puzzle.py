import cv2
import sys
import copy
import heapq
import random
import pyautogui
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') #그래프를 렌더링하는데 필요한 백엔드를 지정하는 코드

imgDic = {1: 'C:/Users/cdh39/PycharmProjects/puzzle/num_1.png',
          2: 'C:/Users/cdh39/PycharmProjects/puzzle/num_2.png',
          3: 'C:/Users/cdh39/PycharmProjects/puzzle/num_3.png',
          4: 'C:/Users/cdh39/PycharmProjects/puzzle/num_4.png',
          5: 'C:/Users/cdh39/PycharmProjects/puzzle/num_5.png',
          6: 'C:/Users/cdh39/PycharmProjects/puzzle/num_6.png',
          7: 'C:/Users/cdh39/PycharmProjects/puzzle/num_7.png',
          8: 'C:/Users/cdh39/PycharmProjects/puzzle/num_8.png',
          0: 'C:/Users/cdh39/PycharmProjects/puzzle/num_0.png'}
def check_zero(puzzle):  # 퍼즐 내에서 0의 위치를 찾아 리턴
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return i, j
def valid_move(x, y): # 0의 위치가 3x3 배열을 벗어날 경우 false, 벗어나지 않을 경우 true 반환
    return 0 <= x < 3 and 0 <= y < 3

def swap(puzzle, x1, y1, x2, y2):  #두 퍼즐을 스왑하는 함수
    puzzle[x1][y1], puzzle[x2][y2] = puzzle[x2][y2], puzzle[x1][y1]

def shuffle_puzzle(puzzle, moves):  #moves 는 스왑 수(몇 번 이동하는지)
    x, y = check_zero(puzzle)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  #상하좌우 이동

    for i in range(moves):
        while True:
            dx, dy = random.choice(directions)   #상하좌우 중 하나 랜덤 선택
            new_x, new_y = x + dx, y + dy        # 0의 좌표 + 상하좌우 중 랜덤 좌표값
            if valid_move(new_x, new_y):         #만약 이동할 0의 위치가 유효한 위치이면
                swap(puzzle, x, y, new_x, new_y) #현재 0이 위치한 퍼즐과 이동할 위치의 퍼즐을 스왑
                x, y = new_x, new_y              #0의 좌표값을 이동한 좌표값으로 갱신
                break

keyList = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]  #목표 퍼즐이자 섞을 퍼즐
goal_puzzle = copy.deepcopy(keyList)         #goal_puzzle은 수동모드 시 정답인지 확인하기 위한 변수

print("목표 퍼즐")
for i in range(3):
    for j in range(3):
        print(keyList[i][j], end=' ')
    print()

shuffle_puzzle(keyList, 100) #keyList에서 0을 moves 번만큼 움직임
print("\n섞은 후의 퍼즐")
for i in range(3):
    for j in range(3):
        print(keyList[i][j], end=' ')
    print()

# 전체그림의 너비와 높이가 각각 5인치인 3x3 서브플롯을 생성(fig는 전체 그림 객체, axes는 서브플롯 객체)
fig, axes = plt.subplots(3, 3, figsize=(5, 5))

# img 는 이미지 파일을 읽어들여 openCV에서 처리할 수 있는 형식으로 로드됨
img1 = cv2.imread(imgDic[keyList[0][0]])
img2 = cv2.imread(imgDic[keyList[0][1]])
img3 = cv2.imread(imgDic[keyList[0][2]])
img4 = cv2.imread(imgDic[keyList[1][0]])
img5 = cv2.imread(imgDic[keyList[1][1]])
img6 = cv2.imread(imgDic[keyList[1][2]])
img7 = cv2.imread(imgDic[keyList[2][0]])
img8 = cv2.imread(imgDic[keyList[2][1]])
img9 = cv2.imread(imgDic[keyList[2][2]])

def print_plot(img1, img2, img3, img4, img5, img6, img7, img8, img9):
    plt.subplot(3, 3, 1)               # 3행 3열의 격자 중 1번째 서브플롯을 선택
    plt.gca().axes.xaxis.set_visible(False)  # x축 눈금과 레이블이 보이지 않게
    plt.gca().axes.yaxis.set_visible(False)  # y축 눈금과 레이블이 보이지 않게
    plt.imshow(img1)                         # img1 이미지를 서브플롯에 출력

    plt.subplot(3, 3, 2)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img2)

    plt.subplot(3, 3, 3)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img3)

    plt.subplot(3, 3, 4)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img4)

    plt.subplot(3, 3, 5)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img5)

    plt.subplot(3, 3, 6)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img6)

    plt.subplot(3, 3, 7)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img7)

    plt.subplot(3, 3, 8)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img8)

    plt.subplot(3, 3, 9)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(img9)

def swap_image(x1, y1, x2, y2): #수동모드 시 사용하는 함수로 퍼즐의 값을 스왑하고 퍼즐이미지도 갱신한다.

    tmp = keyList[x1][y1]
    keyList[x1][y1] = keyList[x2][y2]
    keyList[x2][y2] = tmp

    img1 = cv2.imread(imgDic[keyList[0][0]])
    img2 = cv2.imread(imgDic[keyList[0][1]])
    img3 = cv2.imread(imgDic[keyList[0][2]])
    img4 = cv2.imread(imgDic[keyList[1][0]])
    img5 = cv2.imread(imgDic[keyList[1][1]])
    img6 = cv2.imread(imgDic[keyList[1][2]])
    img7 = cv2.imread(imgDic[keyList[2][0]])
    img8 = cv2.imread(imgDic[keyList[2][1]])
    img9 = cv2.imread(imgDic[keyList[2][2]])
    print_plot(img1, img2, img3, img4, img5, img6, img7, img8, img9)


answer_cnt = 0  #정답을 맞추기까지 걸린 횟수
def draw_puzzle(event):

    global answer_cnt     #전역변수를 수정하기 위해 global 키워드 사용함
    answer_cnt += 1

    if keyList == goal_puzzle:  # 목표 퍼즐에 도달하면 프로그램 종료
        print(f"시도 횟수 : {answer_cnt - 1}")
        print("정답입니다!")
        sys.exit(0)

    if event.button == 1:                   # 왼쪽 마우스가 클릭되면(1)
        fore = pyautogui.getActiveWindow()  # fore에 현재 활성화된 창을 불러오고
        pos = pyautogui.position()          # pos 는 현재 마우스 커서 위치
        x = pos.x - fore.left               # 활성화된 창 내에서 마우스 커서의 x좌표 위치
        y = pos.y - fore.top                # 활성화된 창 내에서 마우스 커서의 y좌표 위치
        print("x좌표 : ", x, ", y좌표 : ", y)

        zero_x, zero_y = check_zero(keyList) #0이 위치한 좌표 확인
        print(f"0 위치 : [{zero_x}][{zero_y}]")

        # 서브플롯 1
        if (x >= 73 and x <= 195) and (y >= 90 and y <= 215):
            if zero_x == 1 and zero_y == 0:
                swap_image(0, 0, 1, 0)
            elif zero_x == 0 and zero_y == 1:
                swap_image(0, 0, 0, 1)

        # 서브플롯 2
        if (x >= 200 and x <= 325) and (y >= 90 and y <= 215):
            if zero_x == 0 and zero_y == 0:
                swap_image(0, 1, 0, 0)
            elif zero_x == 1 and zero_y == 1:
                swap_image(0, 1, 1, 1)
            elif zero_x == 0 and zero_y == 2:
                swap_image(0, 1, 0, 2)
        # 서브플롯 3
        if (x >= 330 and x <= 455) and (y >= 90 and y <= 215):
            if zero_x == 0 and zero_y == 1:
                swap_image(0, 2, 0, 1)
            elif zero_x == 1 and zero_y == 2:
                swap_image(0, 2, 1, 2)

        # 서브플롯 4
        if (x >= 73 and x <= 195) and (y >= 220 and y <= 345):
            if zero_x == 0 and zero_y == 0:
                swap_image(1, 0, 0, 0)
            elif zero_x == 1 and zero_y == 1:
                swap_image(1, 0, 1, 1)
            elif zero_x == 2 and zero_y == 0:
                swap_image(1, 0, 2, 0)

        # 서브플롯 5
        if (x >= 200 and x <= 325) and (y >= 220 and y <= 345):
            if zero_x == 0 and zero_y == 1:
                swap_image(1, 1, 0, 1)
            elif zero_x == 1 and zero_y == 0:
                swap_image(1, 1, 1, 0)
            elif zero_x == 1 and zero_y == 2:
                swap_image(1, 1, 1, 2)
            elif zero_x == 2 and zero_y == 1:
                swap_image(1, 1, 2, 1)
        # 서브플롯 6
        if (x >= 330 and x <= 455) and (y >= 220 and y <= 345):
            if zero_x == 0 and zero_y == 2:
                swap_image(1, 2, 0, 2)
            elif zero_x == 1 and zero_y == 1:
                swap_image(1, 2, 1, 1)
            elif zero_x == 2 and zero_y == 2:
                swap_image(1, 2, 2, 2)

        # 서브플롯 7
        if (x >= 73 and x <= 195) and (y >= 350 and y <= 475):
            if zero_x == 1 and zero_y == 0:
                swap_image(2, 0, 1, 0)
            elif zero_x == 2 and zero_y == 1:
                swap_image(2, 0, 2, 1)

        # 서브플롯 8
        if (x >= 200 and x <= 325) and (y >= 350 and y <= 475):
            if zero_x == 1 and zero_y == 1:
                swap_image(2, 1, 1, 1)
            elif zero_x == 2 and zero_y == 0:
                swap_image(2, 1, 2, 0)
            elif zero_x == 2 and zero_y == 2:
                swap_image(2, 1, 2, 2)

        # 서브플롯 9
        if (x >= 330 and x <= 455) and (y >= 350 and y <= 475):
            if zero_x == 1 and zero_y == 2:
                swap_image(2, 2, 1, 2)
            elif zero_x == 2 and zero_y == 1:
                swap_image(2, 2, 2, 1)
    plt.show()

def auto_puzzle(event): #자동모드 시 실행하는 이벤트 함수
    global keyList
    solution = astar(keyList)

    print(f"시도 횟수 : {len(solution) - 1}번 ")

    for i in solution:
        matrix = i

        img1 = cv2.imread(imgDic[matrix[0][0]])
        img2 = cv2.imread(imgDic[matrix[0][1]])
        img3 = cv2.imread(imgDic[matrix[0][2]])
        img4 = cv2.imread(imgDic[matrix[1][0]])
        img5 = cv2.imread(imgDic[matrix[1][1]])
        img6 = cv2.imread(imgDic[matrix[1][2]])
        img7 = cv2.imread(imgDic[matrix[2][0]])
        img8 = cv2.imread(imgDic[matrix[2][1]])
        img9 = cv2.imread(imgDic[matrix[2][2]])
        print_plot(img1, img2, img3, img4, img5, img6, img7, img8, img9)
        plt.draw()
        plt.pause(1.0)

    print("퍼즐이 완성되었습니다.")
    sys.exit(0)

def h(puzzle, goal): #매개변수로 전달받은 puzzle이 목표퍼즐인 goal과 서로 다른 인덱스 위치 개수 반환
    cnt = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != goal[i][j] and puzzle[i][j] != 0: #0인 경우는 제외(빈칸처리)
                cnt += 1
    return cnt

def f(puzzle, goal, level): # 노드의 레벨과 휴리스틱값 더한 값 반환
    return level + h(puzzle, goal)
def move_puzzle(puzzle, x, y, direction): #퍼즐에서 서로 자리를 바꾸기 위한 함수
    new_puzzle = copy.deepcopy(puzzle)    #객체의 깊은 복사(원본 값이 변경되더라도 영향x)
    if direction == 'up' and x > 0:
        new_puzzle[x][y], new_puzzle[x-1][y] = new_puzzle[x-1][y], new_puzzle[x][y]
    elif direction == 'down' and x < 2:
        new_puzzle[x][y], new_puzzle[x+1][y] = new_puzzle[x+1][y], new_puzzle[x][y]
    elif direction == 'left' and y > 0:
        new_puzzle[x][y], new_puzzle[x][y-1] = new_puzzle[x][y-1], new_puzzle[x][y]
    elif direction == 'right' and y < 2:
        new_puzzle[x][y], new_puzzle[x][y+1] = new_puzzle[x][y+1], new_puzzle[x][y]
    else:
        return None
    return new_puzzle

def astar(start):  # a* 알고리즘 (f함수 값 기준으로 최소힙 사용)
    visit = set()        #노드의 중복을 방지하기 위한 집합 변수
    priority_queue = []  #우선순위 큐
    goal = goal_puzzle   #목표 퍼즐
    oper = ['up', 'down', 'right', 'left']
    node_id = 0  # 각 노드에 고유한 식별자 부여
    #시작 노드 생성
    start_node = {
        'data': start,          #현재 상태의 퍼즐
        'hval': h(start, goal), #시작 노드의 휴리스틱값
        'level': 0,             #현재 노드의 레벨(depth)
        'parent': None,         #현재 노드의 부모 노드
        'id': node_id           #현재 노드의 고유 식별자
    }
    #시작 노드(초기퍼즐)을 우선순위 큐에 삽입
    heapq.heappush(priority_queue, (start_node['hval'], node_id, start_node))
    node_id += 1

    while priority_queue:
        _, _, current = heapq.heappop(priority_queue) # 가장 낮은 f 값을 가진 노드를 꺼냄

        if h(current['data'], goal) == 0:  # 정답과 같으면
            return print_path(current)     # 경로 출력
        else:
            visit.add(tuple(map(tuple, current['data']))) #방문한 노드 기록
            x, y = check_zero(current['data'])            #0의 위치 확인

            for op in oper: #가능한 모든 연산에 반복
                next_puzzle = move_puzzle([row[:] for row in current['data']], x, y, op) #얇은 복사
                #입력받은 퍼즐의 모든 경우의 수 중에서 None 이 아닌 경우와 방문하지 않은 상태면
                if next_puzzle is not None and tuple(map(tuple, next_puzzle)) not in visit:
                    next_node = {
                        'data': next_puzzle,
                        'hval': h(next_puzzle, goal),
                        'level': current['level'] + 1,
                        'parent': current,
                        'id': node_id
                    }
                    #제외할 거 다 제외한 퍼즐을 우선순위 큐에 삽입
                    heapq.heappush(priority_queue, (f(next_node['data'], goal,
                                                    next_node['level']), node_id, next_node))
                    node_id += 1
def print_path(node):   #초기퍼즐부터 목표퍼즐까지 푸는 과정을 반환하는 함수(node 는 목표 퍼즐)
    path = []                     #과정을 담기 위한 리스트
    while node:
        path.append(node['data']) #현재 노드의 데이터를 path 리스트에 추가
        node = node['parent']     #현재 노드를 그 부모 노드로 업데이트(거슬러 올라감)
    return path[::-1]             #path 리스트를 역순으로 뒤집는다.


#########################################################################################

choice = int(input("\n1.수동모드  2.자동모드 중 하나를 선택하세요 : "))

if choice == 1:
    # 맨 처음 화면 초기화
    print_plot(img1, img2, img3, img4, img5, img6, img7, img8, img9)

    # 마우스 이벤트가 발생할 때마다 draw_puzzle 함수 실행
    cid = plt.connect('button_press_event', draw_puzzle)

    plt.subplots_adjust(wspace=0.01, hspace=0.02) # 서브 플롯 간의 여백 조정

    plt.show()  # 플롯을 화면에 표현

elif choice == 2:
    # 맨 처음 화면 초기화
    print_plot(img1, img2, img3, img4, img5, img6, img7, img8, img9)

    # 마우스 이벤트가 발생할 때마다 draw_puzzle 함수 실행
    cid = plt.connect('button_press_event', auto_puzzle)

    plt.subplots_adjust(wspace=0.01, hspace=0.02) # 서브 플롯 간의 여백 조정

    plt.show()  # 플롯을 화면에 표현


