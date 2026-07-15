from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ─── Вложенные Pydantic-модели ────────────────────────────────────────────────
class Address(BaseModel):
    street: str
    city: str
    country: str


class SocialMedia(BaseModel):
    platform: str
    username: str


class CompanyProfile(BaseModel):
    company_name: str = Field(description="Название компании")
    founded_year: int = Field(description="Год основания")
    industry: str = Field(description="Отрасль")
    headquarters: Address = Field(description="Адрес штаб-квартиры")
    ceo: str = Field(description="Имя CEO")
    employee_count: Optional[int] = Field(None, description="Количество сотрудников")
    products: List[str] = Field(description="Список основных продуктов")
    social_media: List[SocialMedia] = Field(description="Социальные сети")


# ─── Запрос ───────────────────────────────────────────────────────────────────
completion = client.beta.chat.completions.parse(
    model="gemma4:latest",
    temperature=0,
    messages=[{"role": "user", "content": "Дай информацию о компании Apple Inc."}],
    response_format=CompanyProfile,
)

profile: CompanyProfile = completion.choices[0].message.parsed

print(f"🏢 Компания:    {profile.company_name}")
print(f"📅 Основана:    {profile.founded_year}")
print(f"🏭 Отрасль:     {profile.industry}")
print(f"📍 Город:       {profile.headquarters.city}, {profile.headquarters.country}")
print(f"👔 CEO:         {profile.ceo}")
print(f"👥 Сотрудники:  {profile.employee_count:,}")
print(f"📦 Продукты:    {', '.join(profile.products)}")
